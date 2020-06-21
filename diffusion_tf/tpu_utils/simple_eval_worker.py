"""
"One-shot" evaluation worker (i.e. run something once, not in a loop over the course of training)

- Computes log prob
- Generates samples progressively
"""

import os
import pickle
import time

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import trange

from .tpu_utils import Model, make_ema, distributed, normalize_data
from .. import utils


def _make_ds_iterator(strategy, ds):
  return strategy.experimental_distribute_dataset(ds).make_initializable_iterator()


class SimpleEvalWorker:
  def __init__(self, tpu_name, model_constructor, total_bs, dataset):
    tf.logging.set_verbosity(tf.logging.INFO)

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
    self.dataset = dataset

    # TPU context
    with self.strategy.scope():
      # Dataset iterators
      self.train_ds_iterator = _make_ds_iterator(
        self.strategy, dataset.train_one_pass_input_fn(params={'batch_size': total_bs}))
      self.eval_ds_iterator = _make_ds_iterator(
        self.strategy, dataset.eval_input_fn(params={'batch_size': total_bs}))

      img_batch_shape = self.train_ds_iterator.output_shapes['image'].as_list()
      assert img_batch_shape[0] == self.local_bs

      # Model
      self.model = model_constructor()
      assert isinstance(self.model, Model)

      # Eval/samples graphs
      print('===== SAMPLES =====')
      self.samples_outputs = self._make_progressive_sampling_graph(img_shape=img_batch_shape[1:])

      # Model with EMA parameters
      print('===== EMA =====')
      self.global_step = tf.train.get_or_create_global_step()
      ema, _ = make_ema(global_step=self.global_step, ema_decay=1e-10, trainable_variables=tf.trainable_variables())

      # EMA versions of the above
      with utils.ema_scope(ema):
        print('===== EMA SAMPLES =====')
        self.ema_samples_outputs = self._make_progressive_sampling_graph(img_shape=img_batch_shape[1:])
        print('===== EMA BPD =====')
        self.bpd_train = self._make_bpd_graph(self.train_ds_iterator)
        self.bpd_eval = self._make_bpd_graph(self.eval_ds_iterator)

  def _make_progressive_sampling_graph(self, img_shape):
    return distributed(
      lambda x_: self.model.progressive_samples_fn(
        x_, tf.random_uniform([self.local_bs], 0, self.dataset.num_classes, dtype=tf.int32)),
      args=(tf.fill([self.local_bs, *img_shape], value=np.nan),),
      reduction='concat', strategy=self.strategy)

  def _make_bpd_graph(self, ds_iterator):
    return distributed(
      lambda x_: self.model.bpd_fn(normalize_data(tf.cast(x_['image'], tf.float32)), x_['label']),
      args=(next(ds_iterator),), reduction='concat', strategy=self.strategy)

  def init_all_iterators(self, sess):
    sess.run([self.train_ds_iterator.initializer, self.eval_ds_iterator.initializer])

  def dump_progressive_samples(self, sess, curr_step, samples_dir, ema: bool, num_samples=50000, batches_per_flush=20):
    if not tf.gfile.IsDirectory(samples_dir):
      tf.gfile.MakeDirs(samples_dir)

    batch_cache, num_flushes_so_far = [], 0

    def _write_batch_cache():
      nonlocal batch_cache, num_flushes_so_far
      # concat all the batches
      assert all(set(b.keys()) == set(self.samples_outputs.keys()) for b in batch_cache)
      concatenated = {
        k: np.concatenate([b[k].astype(np.float32) for b in batch_cache], axis=0)
        for k in self.samples_outputs.keys()
      }
      assert len(set(len(v) for v in concatenated.values())) == 1
      # write the file
      filename = os.path.join(
        samples_dir, 'samples_xstartpred_ema{}_step{:09d}_part{:06d}.pkl'.format(
          int(ema), curr_step, num_flushes_so_far))
      assert not tf.io.gfile.exists(filename), 'samples file already exists: {}'.format(filename)
      print('writing samples batch to:', filename)
      with tf.io.gfile.GFile(filename, 'wb') as f:
        f.write(pickle.dumps(concatenated, protocol=pickle.HIGHEST_PROTOCOL))
      print('done writing samples batch')
      num_flushes_so_far += 1
      batch_cache = []

    num_gen_batches = int(np.ceil(num_samples / self.total_bs))
    print('generating {} samples ({} batches)...'.format(num_samples, num_gen_batches))
    self.init_all_iterators(sess)
    for i_batch in trange(num_gen_batches, desc='sampling'):
      batch_cache.append(sess.run(self.ema_samples_outputs if ema else self.samples_outputs))
      if i_batch != 0 and i_batch % batches_per_flush == 0:
        _write_batch_cache()
    if batch_cache:
      _write_batch_cache()

  def dump_bpd(self, sess, curr_step, output_dir, train: bool, ema: bool):
    assert ema
    if not tf.gfile.IsDirectory(output_dir):
      tf.gfile.MakeDirs(output_dir)
    filename = os.path.join(
      output_dir, 'bpd_{}_ema{}_step{:09d}.pkl'.format('train' if train else 'eval', int(ema), curr_step))
    assert not tf.io.gfile.exists(filename), 'bpd file already exists: {}'.format(filename)
    print('will write bpd data to:', filename)

    batches = []
    tf_op = self.bpd_train if train else self.bpd_eval
    self.init_all_iterators(sess)
    last_print_time = time.time()
    while True:
      try:
        batches.append(sess.run(tf_op))
        if time.time() - last_print_time > 30:
          print('num batches so far: {} ({:.2f} sec)'.format(len(batches), time.time() - last_print_time))
          last_print_time = time.time()
      except tf.errors.OutOfRangeError:
        break

    assert all(set(b.keys()) == set(tf_op.keys()) for b in batches)
    concatenated = {
      k: np.concatenate([b[k].astype(np.float32) for b in batches], axis=0)
      for k in tf_op.keys()
    }
    num_samples = len(list(concatenated.values())[0])
    assert all(len(v) == num_samples for v in concatenated.values())
    print('evaluated on {} examples'.format(num_samples))

    print('writing results to:', filename)
    with tf.io.gfile.GFile(filename, 'wb') as f:
      f.write(pickle.dumps(concatenated, protocol=pickle.HIGHEST_PROTOCOL))
    print('done writing results')

  def run(self, mode: str, logdir: str, load_ckpt: str):
    """
    Main entry point.

    :param mode: what to do
    :param logdir: model directory for the checkpoint to load
    :param load_ckpt: the name of the checkpoint, e.g. "model.ckpt-1000000"
    """

    # Input checkpoint: load_ckpt should be of the form: model.ckpt-1000000
    ckpt = os.path.join(logdir, load_ckpt)
    assert tf.io.gfile.exists(ckpt + '.index')

    # Output dir
    output_dir = os.path.join(logdir, 'simple_eval')
    print('Writing output to: {}'.format(output_dir))

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
      saver.restore(sess, ckpt)
      global_step_val = sess.run(self.global_step)
      print('restored global step: {}'.format(global_step_val))

      if mode in ['bpd_train', 'bpd_eval']:
        self.dump_bpd(
          sess, curr_step=global_step_val, output_dir=os.path.join(output_dir, 'bpd'), ema=True,
          train=mode == 'bpd_train')
      elif mode == 'progressive_samples':
        return self.dump_progressive_samples(
          sess, curr_step=global_step_val, samples_dir=os.path.join(output_dir, 'progressive_samples'), ema=True)
      else:
        raise NotImplementedError(mode)