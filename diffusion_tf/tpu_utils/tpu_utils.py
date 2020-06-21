import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.tpu import tpu_function
from tqdm import trange

from . import classifier_metrics_numpy
from .tpu_summaries import TpuSummaries
from .. import utils


# ========== TPU utilities ==========

def num_tpu_replicas():
  return tpu_function.get_tpu_context().number_of_shards


def get_tpu_replica_id():
  with tf.control_dependencies(None):
    return tpu_ops.tpu_replicated_input(list(range(num_tpu_replicas())))


def distributed(fn, *, args, reduction, strategy):
  """
  Sharded computation followed by concat/mean for TPUStrategy.
  """
  out = strategy.experimental_run_v2(fn, args=args)
  if reduction == 'mean':
    return tf.nest.map_structure(lambda x: tf.reduce_mean(strategy.reduce('mean', x)), out)
  elif reduction == 'concat':
    return tf.nest.map_structure(lambda x: tf.concat(strategy.experimental_local_results(x), axis=0), out)
  else:
    raise NotImplementedError(reduction)


# ========== Inception utilities ==========

INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05_v4.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score_tpu.pb'
INCEPTION_GRAPH_DEF = tfgan.eval.get_graph_def_from_url_tarball(
  INCEPTION_URL, INCEPTION_FROZEN_GRAPH, os.path.basename(INCEPTION_URL))


def run_inception(images):
  assert images.dtype == tf.float32  # images should be in [-1, 1]
  out = tfgan.eval.run_inception(
    images,
    graph_def=INCEPTION_GRAPH_DEF,
    default_graph_def_fn=None,
    output_tensor=['pool_3:0', 'logits:0']
  )
  return {'pool_3': out[0], 'logits': out[1]}


# ========== Training ==========

normalize_data = lambda x_: x_ / 127.5 - 1.
unnormalize_data = lambda x_: (x_ + 1.) * 127.5


class Model:
  # All images (inputs and outputs) should be normalized to [-1, 1]
  def train_fn(self, x, y) -> dict:
    raise NotImplementedError

  def samples_fn(self, dummy_x, y) -> dict:
    raise NotImplementedError

  def sample_and_run_inception(self, dummy_x, y, clip_samples=True):
    samples_dict = self.samples_fn(dummy_x, y)
    assert isinstance(samples_dict, dict)
    return {
      k: run_inception(tfgan.eval.preprocess_image(unnormalize_data(
        tf.clip_by_value(v, -1., 1.) if clip_samples else v)))
      for (k, v) in samples_dict.items()
    }

  def bpd_fn(self, x, y) -> dict:
    return None


def make_ema(global_step, ema_decay, trainable_variables):
  ema = tf.train.ExponentialMovingAverage(decay=tf.where(tf.less(global_step, 1), 1e-10, ema_decay))
  ema_op = ema.apply(trainable_variables)
  return ema, ema_op


def load_train_kwargs(model_dir):
  with tf.io.gfile.GFile(os.path.join(model_dir, 'kwargs.json'), 'r') as f:
    kwargs = json.loads(f.read())
  return kwargs


def run_training(
    *, model_constructor, train_input_fn, total_bs,
    optimizer, lr, warmup, grad_clip, ema_decay=0.9999,
    tpu=None, zone=None, project=None,
    log_dir, exp_name, dump_kwargs=None,
    date_str=None, iterations_per_loop=1000, keep_checkpoint_max=2, max_steps=int(1e10),
    warm_start_from=None
):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Create checkpoint directory
  model_dir = os.path.join(
    log_dir,
    datetime.now().strftime('%Y-%m-%d') if date_str is None else date_str,
    exp_name
  )
  print('model dir:', model_dir)
  if tf.io.gfile.exists(model_dir):
    print('model dir already exists: {}'.format(model_dir))
    if input('continue training? [y/n] ') != 'y':
      print('aborting')
      return

  # Save kwargs in json format
  if dump_kwargs is not None:
    with tf.io.gfile.GFile(os.path.join(model_dir, 'kwargs.json'), 'w') as f:
      f.write(json.dumps(dump_kwargs, indent=2, sort_keys=True) + '\n')

  # model_fn for TPUEstimator
  def model_fn(features, params, mode):
    local_bs = params['batch_size']
    print('Global batch size: {}, local batch size: {}'.format(total_bs, local_bs))
    assert total_bs == num_tpu_replicas() * local_bs

    assert mode == tf.estimator.ModeKeys.TRAIN, 'only TRAIN mode supported'
    assert features['image'].shape[0] == local_bs
    assert features['label'].shape == [local_bs] and features['label'].dtype == tf.int32
    # assert labels.dtype == features['label'].dtype and labels.shape == features['label'].shape

    del params

    ##########

    # create model
    model = model_constructor()
    assert isinstance(model, Model)

    # training loss
    train_info_dict = model.train_fn(normalize_data(tf.cast(features['image'], tf.float32)), features['label'])
    loss = train_info_dict['loss']
    assert loss.shape == []

    # train op
    trainable_variables = tf.trainable_variables()
    print('num params: {:,}'.format(sum(int(np.prod(p.shape.as_list())) for p in trainable_variables)))
    global_step = tf.train.get_or_create_global_step()
    warmed_up_lr = utils.get_warmed_up_lr(max_lr=lr, warmup=warmup, global_step=global_step)
    train_op, gnorm = utils.make_optimizer(
      loss=loss,
      trainable_variables=trainable_variables,
      global_step=global_step,
      lr=warmed_up_lr,
      optimizer=optimizer,
      grad_clip=grad_clip / float(num_tpu_replicas()),
      tpu=True
    )

    # ema
    ema, ema_op = make_ema(global_step=global_step, ema_decay=ema_decay, trainable_variables=trainable_variables)
    with tf.control_dependencies([train_op]):
      train_op = tf.group(ema_op)

    # summary
    tpu_summary = TpuSummaries(model_dir, save_summary_steps=100)
    tpu_summary.scalar('train/loss', loss)
    tpu_summary.scalar('train/gnorm', gnorm)
    tpu_summary.scalar('train/pnorm', utils.rms(trainable_variables))
    tpu_summary.scalar('train/lr', warmed_up_lr)
    return tf.estimator.tpu.TPUEstimatorSpec(
      mode=mode, host_call=tpu_summary.get_host_call(), loss=loss, train_op=train_op)

  # Set up Estimator and train
  print("warm_start_from:", warm_start_from)
  estimator = tf.estimator.tpu.TPUEstimator(
    model_fn=model_fn,
    use_tpu=True,
    train_batch_size=total_bs,
    eval_batch_size=total_bs,
    config=tf.estimator.tpu.RunConfig(
      cluster=tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu, zone=zone, project=project),
      model_dir=model_dir,
      session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.estimator.tpu.TPUConfig(
        iterations_per_loop=iterations_per_loop,
        num_shards=None,
        per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
      ),
      save_checkpoints_secs=1600,  # 30 minutes
      keep_checkpoint_max=keep_checkpoint_max
    ),
    warm_start_from=warm_start_from
  )
  estimator.train(input_fn=train_input_fn, max_steps=max_steps)


# ========== Evaluation / sampling ==========


class InceptionFeatures:
  """
  Compute and store Inception features for a dataset
  """

  def __init__(self, dataset, strategy, limit_dataset_size=0):
    # distributed dataset iterator
    if limit_dataset_size > 0:
      dataset = dataset.take(limit_dataset_size)
    self.ds_iterator = strategy.experimental_distribute_dataset(dataset).make_initializable_iterator()

    # inception network on the dataset
    self.inception_real = distributed(
      lambda x_: run_inception(tfgan.eval.preprocess_image(x_['image'])),
      args=(next(self.ds_iterator),), reduction='concat', strategy=strategy)

    self.cached_inception_real = None  # cached inception features
    self.real_inception_score = None  # saved inception scores for the dataset

  def get(self, sess):
    # On the first invocation, compute Inception activations for the eval dataset
    if self.cached_inception_real is None:
      print('computing inception features on the eval set...')
      sess.run(self.ds_iterator.initializer)  # reset the eval dataset iterator
      inception_real_batches, tstart = [], time.time()
      while True:
        try:
          inception_real_batches.append(sess.run(self.inception_real))
        except tf.errors.OutOfRangeError:
          break
      self.cached_inception_real = {
        feat_key: np.concatenate([batch[feat_key] for batch in inception_real_batches], axis=0).astype(np.float64)
        for feat_key in ['pool_3', 'logits']
      }
      print('cached eval inception tensors: logits: {}, pool_3: {} (time: {})'.format(
        self.cached_inception_real['logits'].shape, self.cached_inception_real['pool_3'].shape,
        time.time() - tstart))

      self.real_inception_score = float(
        classifier_metrics_numpy.classifier_score_from_logits(self.cached_inception_real['logits']))
      del self.cached_inception_real['logits']  # save memory
    print('real inception score', self.real_inception_score)

    return self.cached_inception_real, self.real_inception_score


class EvalWorker:
  def __init__(self, tpu_name, model_constructor, total_bs, dataset, inception_bs=8, num_inception_samples=1024, limit_dataset_size=0):

    self.resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.tpu.experimental.initialize_tpu_system(self.resolver)
    self.strategy = tf.distribute.experimental.TPUStrategy(self.resolver)

    self.num_cores = self.strategy.num_replicas_in_sync
    assert total_bs % self.num_cores == 0
    self.total_bs = total_bs
    self.local_bs = total_bs // self.num_cores
    print('num cores: {}'.format(self.num_cores))
    print('total batch size: {}'.format(self.total_bs))
    print('local batch size: {}'.format(self.local_bs))
    self.num_inception_samples = num_inception_samples
    assert inception_bs % self.num_cores == 0
    self.inception_bs = inception_bs
    self.inception_local_bs = inception_bs // self.num_cores
    self.dataset = dataset
    assert dataset.num_classes == 1, 'not supported'

    # TPU context
    with self.strategy.scope():
      # Inception network on real data
      print('===== INCEPTION =====')
      # Eval dataset iterator (this is the training set without repeat & shuffling)
      self.inception_real_train = InceptionFeatures(
        dataset=dataset.train_one_pass_input_fn(params={'batch_size': total_bs}), strategy=self.strategy, limit_dataset_size=limit_dataset_size // total_bs)
      # Val dataset, if it exists
      val_ds = dataset.eval_input_fn(params={'batch_size': total_bs})
      self.inception_real_val = None if val_ds is None else InceptionFeatures(dataset=val_ds, strategy=self.strategy, limit_dataset_size=limit_dataset_size // total_bs)

      img_batch_shape = self.inception_real_train.ds_iterator.output_shapes['image'].as_list()
      assert img_batch_shape[0] == self.local_bs

      # Model
      self.model = model_constructor()
      assert isinstance(self.model, Model)

      # Eval/samples graphs
      print('===== SAMPLES =====')
      self.samples_outputs, self.samples_inception = self._make_sampling_graph(
        img_shape=img_batch_shape[1:], with_inception=True)

      # Model with EMA parameters
      self.global_step = tf.train.get_or_create_global_step()
      print('===== EMA =====')
      ema, _ = make_ema(global_step=self.global_step, ema_decay=1e-10, trainable_variables=tf.trainable_variables())

      # EMA versions of the above
      with utils.ema_scope(ema):
        print('===== EMA SAMPLES =====')
        self.ema_samples_outputs, self.ema_samples_inception = self._make_sampling_graph(
          img_shape=img_batch_shape[1:], with_inception=True)

  def _make_sampling_graph(self, img_shape, with_inception):

    def _make_inputs(total_bs, local_bs):
      # Dummy inputs to feed to samplers
      input_x = tf.fill([local_bs, *img_shape], value=np.nan)
      input_y = tf.random_uniform([local_bs], 0, self.dataset.num_classes, dtype=tf.int32)
      return input_x, input_y

    # Samples
    samples_outputs = distributed(
      self.model.samples_fn,
      args=_make_inputs(self.total_bs, self.local_bs),
      reduction='concat', strategy=self.strategy)
    if not with_inception:
      return samples_outputs

    # Inception activations of samples
    samples_inception = distributed(
      self.model.sample_and_run_inception,
      args=_make_inputs(self.inception_bs, self.inception_local_bs),
      reduction='concat', strategy=self.strategy)
    return samples_outputs, samples_inception

  def _run_sampling(self, sess, ema: bool):
    out = {}
    print('sampling...')
    tstart = time.time()
    samples = sess.run(self.ema_samples_outputs if ema else self.samples_outputs)
    print('sampling done in {} sec'.format(time.time() - tstart))
    for k, v in samples.items():
      out['samples/{}'.format(k)] = v
    return out

  def _run_metrics(self, sess, ema: bool):
    print('computing sample quality metrics...')
    metrics = {}

    # Get Inception activations on the real dataset
    cached_inception_real_train, metrics['real_inception_score_train'] = self.inception_real_train.get(sess)
    if self.inception_real_val is not None:
      cached_inception_real_val, metrics['real_inception_score'] = self.inception_real_val.get(sess)
    else:
      cached_inception_real_val = None

    # Generate lots of samples
    num_inception_gen_batches = int(np.ceil(self.num_inception_samples / self.inception_bs))
    print('generating {} samples and inception features ({} batches)...'.format(
      self.num_inception_samples, num_inception_gen_batches))
    inception_gen_batches = [
      sess.run(self.ema_samples_inception if ema else self.samples_inception)
      for _ in trange(num_inception_gen_batches, desc='sampling inception batch')
    ]

    # Compute FID and Inception score
    assert set(self.samples_outputs.keys()) == set(inception_gen_batches[0].keys())
    for samples_key in self.samples_outputs.keys():
      # concat features from all batches into a single array
      inception_gen = {
        feat_key: np.concatenate(
          [batch[samples_key][feat_key] for batch in inception_gen_batches], axis=0
        )[:self.num_inception_samples].astype(np.float64)
        for feat_key in ['pool_3', 'logits']
      }
      assert all(v.shape[0] == self.num_inception_samples for v in inception_gen.values())

      # Inception score
      metrics['{}/inception{}'.format(samples_key, self.num_inception_samples)] = float(
        classifier_metrics_numpy.classifier_score_from_logits(inception_gen['logits']))

      # FID vs training set
      metrics['{}/trainfid{}'.format(samples_key, self.num_inception_samples)] = float(
        classifier_metrics_numpy.frechet_classifier_distance_from_activations(
          cached_inception_real_train['pool_3'], inception_gen['pool_3']))

      # FID vs val set
      if cached_inception_real_val is not None:
        metrics['{}/fid{}'.format(samples_key, self.num_inception_samples)] = float(
          classifier_metrics_numpy.frechet_classifier_distance_from_activations(
            cached_inception_real_val['pool_3'], inception_gen['pool_3']))

    return metrics

  def _write_eval_and_samples(self, sess, log: utils.SummaryWriter, curr_step, prefix, ema: bool):
    # Samples
    for k, v in self._run_sampling(sess, ema=ema).items():
      assert len(v.shape) == 4 and v.shape[0] == self.total_bs
      log.images('{}/{}'.format(prefix, k), np.clip(unnormalize_data(v), 0, 255).astype('uint8'), step=curr_step)
    log.flush()

    # Metrics
    metrics = self._run_metrics(sess, ema=ema)
    print('metrics:', json.dumps(metrics, indent=2, sort_keys=True))
    for k, v in metrics.items():
      log.scalar('{}/{}'.format(prefix, k), v, step=curr_step)
    log.flush()

  def _dump_samples(self, sess, curr_step, samples_dir, ema: bool, num_samples=50000):
    print('will dump samples to', samples_dir)
    if not tf.gfile.IsDirectory(samples_dir):
      tf.gfile.MakeDirs(samples_dir)
    filename = os.path.join(
      samples_dir, 'samples_ema{}_step{:09d}.pkl'.format(int(ema), curr_step))
    assert not tf.io.gfile.exists(filename), 'samples file already exists: {}'.format(filename)

    num_gen_batches = int(np.ceil(num_samples / self.total_bs))
    print('generating {} samples ({} batches)...'.format(num_samples, num_gen_batches))

    # gen_batches = [
    #   sess.run(self.ema_samples_outputs if ema else self.samples_outputs)
    #   for _ in trange(num_gen_batches, desc='sampling')
    # ]
    # assert all(set(b.keys()) == set(self.samples_outputs.keys()) for b in gen_batches)
    # concatenated = {
    #   k: np.concatenate([b[k].astype(np.float32) for b in gen_batches], axis=0)[:num_samples]
    #   for k in self.samples_outputs.keys()
    # }
    # assert all(len(v) == num_samples for v in concatenated.values())
    #
    # print('writing samples to:', filename)
    # with tf.io.gfile.GFile(filename, 'wb') as f:
    #   f.write(pickle.dumps(concatenated, protocol=pickle.HIGHEST_PROTOCOL))

    for i in trange(num_gen_batches, desc='sampling'):
        b = sess.run(self.ema_samples_outputs if ema else self.samples_outputs)
        assert set(b.keys()) == set(self.samples_outputs.keys())
        b = {
          k: b[k].astype(np.float32) for k in self.samples_outputs.keys()
        }
        #assert all(len(v) == num_samples for v in concatenated.values())

        filename_i = "{}.batch{:05d}".format(filename, i)
        print('writing samples for batch', i, 'to:', filename_i)
        with tf.io.gfile.GFile(filename_i, 'wb') as f:
            f.write(pickle.dumps(b, protocol=pickle.HIGHEST_PROTOCOL))
    print('done writing samples')

  def run(self, logdir, once: bool, skip_non_ema_pass=True, dump_samples_only=False, load_ckpt=None, samples_dir=None, seed=0):
    """Runs the eval/sampling worker loop.
    Args:
      logdir: directory to read checkpoints from
      once: if True, writes results to a temporary directory (not to logdir),
        and exits after evaluating one checkpoint.
    """
    tf.logging.set_verbosity(tf.logging.INFO)

    # Are we evaluating a single checkpoint or looping on the latest?
    if load_ckpt is not None:
      # load_ckpt should be of the form: model.ckpt-1000000
      assert tf.io.gfile.exists(os.path.join(logdir, load_ckpt) + '.index')
      ckpt_iterator = [os.path.join(logdir, load_ckpt)]  # load this one checkpoint only
    else:
      ckpt_iterator = tf.train.checkpoints_iterator(logdir)  # wait for checkpoints to come in
    assert tf.io.gfile.isdir(logdir), 'expected {} to be a directory'.format(logdir)

    # Set up eval SummaryWriter
    if once:
      eval_logdir = os.path.join(logdir, 'eval_once_{}'.format(time.time()))
    else:
      eval_logdir = os.path.join(logdir, 'eval')
    print('Writing eval data to: {}'.format(eval_logdir))
    eval_log = utils.SummaryWriter(eval_logdir, write_graph=False)

    # Make the session
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    cluster_spec = self.resolver.cluster_spec()
    if cluster_spec:
      config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
    print('making session...')
    with tf.Session(target=self.resolver.master(), config=config) as sess:

      print('initializing global variables')
      sess.run(tf.global_variables_initializer())

      # Checkpoint loading
      print('making saver')
      saver = tf.train.Saver()

      for ckpt in ckpt_iterator:
        # Restore params
        saver.restore(sess, ckpt)
        global_step_val = sess.run(self.global_step)
        print('restored global step: {}'.format(global_step_val))

        print('seeding')
        utils.seed_all(seed)

        print('ema pass')
        if dump_samples_only:
          if not samples_dir:
            samples_dir = os.path.join(eval_logdir, '{}_samples{}'.format(type(self.dataset).__name__, global_step_val))
          self._dump_samples(
            sess, curr_step=global_step_val, samples_dir=samples_dir, ema=True)
        else:
          self._write_eval_and_samples(sess, log=eval_log, curr_step=global_step_val, prefix='eval_ema', ema=True)

        if not skip_non_ema_pass:
          print('non-ema pass')
          if dump_samples_only:
            self._dump_samples(
              sess, curr_step=global_step_val, samples_dir=os.path.join(eval_logdir, 'samples'), ema=False)
          else:
            self._write_eval_and_samples(sess, log=eval_log, curr_step=global_step_val, prefix='eval', ema=False)

        if once:
          break
