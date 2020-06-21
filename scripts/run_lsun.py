"""
LSUN church, bedroom and cat 256x256

# LSUN church
python3 scripts/run_lsun.py train --bucket_name_prefix $BUCKET_PREFIX --tpu_name $TPU_NAME --exp_name $EXPERIMENT_NAME --tpu_name $TPU_NAME --tfr_file 'tensorflow_datasets/lsun/church/church-r08.tfrecords'
python3 scripts/run_lsun.py evaluation --tpu_name $EVAL_TPU_NAME --model_dir $MODEL_DIR --tfr_file 'tensorflow_datasets/lsun/church/church-r08.tfrecords'

# LSUN bedroom
python3 scripts/run_lsun.py train --bucket_name_prefix $BUCKET_PREFIX --exp_name $EXPERIMENT_NAME --tfr_file 'tensorflow_datasets/lsun/bedroom-full/bedroom-full-r08.tfrecords'
python3 scripts/run_lsun.py evaluation --bucket_name_prefix $BUCKET_PREFIX --tpu_name $EVAL_TPU_NAME --model_dir $MODEL_DIR --tfr_file 'tensorflow_datasets/lsun/bedroom-full/bedroom-full-r08.tfrecords'

# LSUN cat
python3 scripts/run_lsun.py train --bucket_name_prefix $BUCKET_PREFIX --exp_name $EXPERIMENT_NAME --tpu_name $TPU_NAME --tfr_file 'tensorflow_datasets/lsun/cat/cat-r08.tfrecords' --randflip 0
python3 scripts/run_lsun.py evaluation --bucket_name_prefix $BUCKET_PREFIX --tpu_name $EVAL_TPU_NAME --model_dir $MODEL_DIR --tfr_file 'tensorflow_datasets/lsun/cat/cat-r08.tfrecords'
"""

import functools

import fire
import numpy as np
import tensorflow.compat.v1 as tf

from diffusion_tf import utils
from diffusion_tf.diffusion_utils import get_beta_schedule, GaussianDiffusion
from diffusion_tf.models import unet
from diffusion_tf.tpu_utils import tpu_utils, datasets


class Model(tpu_utils.Model):
  def __init__(self, *, model_name, betas: np.ndarray, loss_type: str, num_classes: int,
               dropout: float, randflip, block_size: int):
    self.model_name = model_name
    self.diffusion = GaussianDiffusion(betas=betas, loss_type=loss_type)
    self.num_classes = num_classes
    self.dropout = dropout
    self.randflip = randflip
    self.block_size = block_size

  def _denoise(self, x, t, y, dropout):
    B, H, W, C = x.shape.as_list()
    assert x.dtype == tf.float32
    assert t.shape == [B] and t.dtype in [tf.int32, tf.int64]
    assert y.shape == [B] and y.dtype in [tf.int32, tf.int64]
    orig_out_ch = out_ch = C

    if self.block_size != 1:
      x = tf.nn.space_to_depth(x, self.block_size)
      out_ch *= self.block_size ** 2

    y = None
    if self.model_name == 'unet2d16b2c112244':  # 114M for block_size=1
      out = unet.model(
        x, t=t, y=y, name='model', ch=128, ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2, attn_resolutions=(16,),
        out_ch=out_ch, num_classes=self.num_classes, dropout=dropout
      )
    else:
      raise NotImplementedError(self.model_name)

    if self.block_size != 1:
      out = tf.nn.depth_to_space(out, self.block_size)
    assert out.shape == [B, H, W, orig_out_ch]
    return out

  def train_fn(self, x, y):
    B, H, W, C = x.shape
    if self.randflip:
      x = tf.image.random_flip_left_right(x)
      assert x.shape == [B, H, W, C]
    t = tf.random_uniform([B], 0, self.diffusion.num_timesteps, dtype=tf.int32)
    losses = self.diffusion.p_losses(
      denoise_fn=functools.partial(self._denoise, y=y, dropout=self.dropout), x_start=x, t=t)
    assert losses.shape == t.shape == [B]
    return {'loss': tf.reduce_mean(losses)}

  def samples_fn(self, dummy_noise, y):
    return {
      'samples': self.diffusion.p_sample_loop(
        denoise_fn=functools.partial(self._denoise, y=y, dropout=0),
        shape=dummy_noise.shape.as_list(),
        noise_fn=tf.random_normal
      )
    }


def evaluation(
    model_dir, tpu_name, bucket_name_prefix, once=False, dump_samples_only=False, total_bs=128,
    tfr_file='tensorflow_datasets/lsun/church-r08.tfrecords', samples_dir=None, num_inception_samples=2048,
):
  region = utils.get_gcp_region()
  tfr_file = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfr_file)
  kwargs = tpu_utils.load_train_kwargs(model_dir)
  print('loaded kwargs:', kwargs)
  ds = datasets.get_dataset(kwargs['dataset'], tfr_file=tfr_file)
  worker = tpu_utils.EvalWorker(
    tpu_name=tpu_name,
    model_constructor=lambda: Model(
      model_name=kwargs['model_name'],
      betas=get_beta_schedule(
        kwargs['beta_schedule'], beta_start=kwargs['beta_start'], beta_end=kwargs['beta_end'],
        num_diffusion_timesteps=kwargs['num_diffusion_timesteps']
      ),
      loss_type=kwargs['loss_type'],
      num_classes=ds.num_classes,
      dropout=kwargs['dropout'],
      randflip=kwargs['randflip'],
      block_size=kwargs['block_size']
    ),
    total_bs=total_bs, inception_bs=total_bs, num_inception_samples=num_inception_samples,
    dataset=ds,
    limit_dataset_size=30000  # limit size of dataset for computing Inception features, for memory reasons
  )
  worker.run(logdir=model_dir, once=once, skip_non_ema_pass=True, dump_samples_only=dump_samples_only,
             samples_dir=samples_dir)


def train(
    exp_name, tpu_name, bucket_name_prefix, model_name='unet2d16b2c112244', dataset='lsun',
    optimizer='adam', total_bs=64, grad_clip=1., lr=2e-5, warmup=5000,
    num_diffusion_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule='linear', loss_type='noisepred',
    dropout=0.0, randflip=1, block_size=1,
    tfr_file='tensorflow_datasets/lsun/church/church-r08.tfrecords', log_dir='logs',
    warm_start_model_dir=None
):
  region = utils.get_gcp_region()
  tfr_file = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, tfr_file)
  log_dir = 'gs://{}-{}/{}'.format(bucket_name_prefix, region, log_dir)
  print("tfr_file:", tfr_file)
  print("log_dir:", log_dir)
  kwargs = dict(locals())
  ds = datasets.get_dataset(dataset, tfr_file=tfr_file)
  tpu_utils.run_training(
    date_str='9999-99-99',
    exp_name='{exp_name}_{dataset}_{model_name}_{optimizer}_bs{total_bs}_lr{lr}w{warmup}_beta{beta_start}-{beta_end}-{beta_schedule}_t{num_diffusion_timesteps}_{loss_type}_dropout{dropout}_randflip{randflip}_blk{block_size}'.format(
      **kwargs),
    model_constructor=lambda: Model(
      model_name=model_name,
      betas=get_beta_schedule(
        beta_schedule, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=num_diffusion_timesteps
      ),
      loss_type=loss_type,
      num_classes=ds.num_classes,
      dropout=dropout,
      randflip=randflip,
      block_size=block_size
    ),
    optimizer=optimizer, total_bs=total_bs, lr=lr, warmup=warmup, grad_clip=grad_clip,
    train_input_fn=ds.train_input_fn,
    tpu=tpu_name, log_dir=log_dir, dump_kwargs=kwargs,
    warm_start_from=tf.estimator.WarmStartSettings(
      ckpt_to_initialize_from=tf.train.latest_checkpoint(warm_start_model_dir),
      vars_to_warm_start=[".*"]
    ) if warm_start_model_dir else None
  )


if __name__ == '__main__':
  fire.Fire()
