import contextlib
import io
import random
import time

import numpy as np
import tensorflow.compat.v1 as tf
from PIL import Image
from tensorflow.compat.v1 import gfile
from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.util.event_pb2 import Event


class SummaryWriter:
  """Tensorflow summary writer inspired by Jaxboard.
  This version doesn't try to avoid Tensorflow dependencies, because this
  project uses Tensorflow.
  """

  def __init__(self, dir, write_graph=True):
    if not gfile.IsDirectory(dir):
      gfile.MakeDirs(dir)
    self.writer = tf.summary.FileWriter(
      dir, graph=tf.get_default_graph() if write_graph else None)

  def flush(self):
    self.writer.flush()

  def close(self):
    self.writer.close()

  def _write_event(self, summary_value, step):
    self.writer.add_event(
      Event(
        wall_time=round(time.time()),
        step=step,
        summary=Summary(value=[summary_value])))

  def scalar(self, tag, value, step):
    self._write_event(Summary.Value(tag=tag, simple_value=float(value)), step)

  def image(self, tag, image, step):
    image = np.asarray(image)
    if image.ndim == 2:
      image = image[:, :, None]
    if image.shape[-1] == 1:
      image = np.repeat(image, 3, axis=-1)

    bytesio = io.BytesIO()
    Image.fromarray(image).save(bytesio, 'PNG')
    image_summary = Summary.Image(
      encoded_image_string=bytesio.getvalue(),
      colorspace=3,
      height=image.shape[0],
      width=image.shape[1])
    self._write_event(Summary.Value(tag=tag, image=image_summary), step)

  def images(self, tag, images, step):
    self.image(tag, tile_imgs(images), step=step)


def seed_all(seed):
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)


def tile_imgs(imgs, *, pad_pixels=1, pad_val=255, num_col=0):
  assert pad_pixels >= 0 and 0 <= pad_val <= 255

  imgs = np.asarray(imgs)
  assert imgs.dtype == np.uint8
  if imgs.ndim == 3:
    imgs = imgs[..., None]
  n, h, w, c = imgs.shape
  assert c == 1 or c == 3, 'Expected 1 or 3 channels'

  if num_col <= 0:
    # Make a square
    ceil_sqrt_n = int(np.ceil(np.sqrt(float(n))))
    num_row = ceil_sqrt_n
    num_col = ceil_sqrt_n
  else:
    # Make a B/num_per_row x num_per_row grid
    assert n % num_col == 0
    num_row = int(np.ceil(n / num_col))

  imgs = np.pad(
    imgs,
    pad_width=((0, num_row * num_col - n), (pad_pixels, pad_pixels), (pad_pixels, pad_pixels), (0, 0)),
    mode='constant',
    constant_values=pad_val
  )
  h, w = h + 2 * pad_pixels, w + 2 * pad_pixels
  imgs = imgs.reshape(num_row, num_col, h, w, c)
  imgs = imgs.transpose(0, 2, 1, 3, 4)
  imgs = imgs.reshape(num_row * h, num_col * w, c)

  if pad_pixels > 0:
    imgs = imgs[pad_pixels:-pad_pixels, pad_pixels:-pad_pixels, :]
  if c == 1:
    imgs = imgs[..., 0]
  return imgs


def save_tiled_imgs(filename, imgs, pad_pixels=1, pad_val=255, num_col=0):
  Image.fromarray(tile_imgs(imgs, pad_pixels=pad_pixels, pad_val=pad_val, num_col=num_col)).save(filename)


# ===

def approx_standard_normal_cdf(x):
  return 0.5 * (1.0 + tf.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
  # Assumes data is integers [0, 255] rescaled to [-1, 1]
  assert x.shape == means.shape == log_scales.shape
  centered_x = x - means
  inv_stdv = tf.exp(-log_scales)
  plus_in = inv_stdv * (centered_x + 1. / 255.)
  cdf_plus = approx_standard_normal_cdf(plus_in)
  min_in = inv_stdv * (centered_x - 1. / 255.)
  cdf_min = approx_standard_normal_cdf(min_in)
  log_cdf_plus = tf.log(tf.maximum(cdf_plus, 1e-12))
  log_one_minus_cdf_min = tf.log(tf.maximum(1. - cdf_min, 1e-12))
  cdf_delta = cdf_plus - cdf_min
  log_probs = tf.where(
    x < -0.999, log_cdf_plus,
    tf.where(x > 0.999, log_one_minus_cdf_min,
             tf.log(tf.maximum(cdf_delta, 1e-12))))
  assert log_probs.shape == x.shape
  return log_probs


# ===


def rms(variables):
  return tf.sqrt(
    sum([tf.reduce_sum(tf.square(v)) for v in variables]) /
    sum(int(np.prod(v.shape.as_list())) for v in variables))


def get_warmed_up_lr(max_lr, warmup, global_step):
  if warmup == 0:
    return max_lr
  return max_lr * tf.minimum(tf.cast(global_step, tf.float32) / float(warmup), 1.0)


def make_optimizer(
    *,
    loss, trainable_variables, global_step, tpu: bool,
    optimizer: str, lr: float, grad_clip: float,
    rmsprop_decay=0.95, rmsprop_momentum=0.9, epsilon=1e-8
):
  if optimizer == 'adam':
    optimizer = tf.train.AdamOptimizer(
      learning_rate=lr, epsilon=epsilon)
  elif optimizer == 'rmsprop':
    optimizer = tf.train.RMSPropOptimizer(
      learning_rate=lr, decay=rmsprop_decay, momentum=rmsprop_momentum, epsilon=epsilon)
  else:
    raise NotImplementedError(optimizer)

  if tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  # compute gradient
  grads_and_vars = optimizer.compute_gradients(loss, var_list=trainable_variables)

  # clip gradient
  clipped_grads, gnorm = tf.clip_by_global_norm([g for (g, _) in grads_and_vars], grad_clip)
  grads_and_vars = [(g, v) for g, (_, v) in zip(clipped_grads, grads_and_vars)]

  # train
  train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
  return train_op, gnorm


@contextlib.contextmanager
def ema_scope(orig_model_ema):
  def _ema_getter(getter, name, *args, **kwargs):
    v = getter(name, *args, **kwargs)
    v = orig_model_ema.average(v)
    if v is None:
      raise RuntimeError('Variable {} has no EMA counterpart'.format(name))
    return v

  with tf.variable_scope(tf.get_variable_scope(), custom_getter=_ema_getter, reuse=True):
    with tf.name_scope('ema_scope'):
      yield


def get_gcp_region():
  # https://stackoverflow.com/a/31689692
  import requests
  metadata_server = "http://metadata/computeMetadata/v1/instance/"
  metadata_flavor = {'Metadata-Flavor': 'Google'}
  zone = requests.get(metadata_server + 'zone', headers=metadata_flavor).text
  zone = zone.split('/')[-1]
  region = '-'.join(zone.split('-')[:-1])
  return region
