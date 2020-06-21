import math
import string

import tensorflow.compat.v1 as tf

# ===== Neural network building defaults =====
DEFAULT_DTYPE = tf.float32


def default_init(scale):
  return tf.initializers.variance_scaling(scale=1e-10 if scale == 0 else scale, mode='fan_avg', distribution='uniform')


# ===== Utilities =====

def _wrapped_print(x, *args, **kwargs):
  print_op = tf.print(*args, **kwargs)
  with tf.control_dependencies([print_op]):
    return tf.identity(x)


def debug_print(x, name):
  return _wrapped_print(x, name, tf.reduce_mean(x), tf.math.reduce_std(x), tf.reduce_min(x), tf.reduce_max(x))


def flatten(x):
  return tf.reshape(x, [int(x.shape[0]), -1])


def sumflat(x):
  return tf.reduce_sum(x, axis=list(range(1, len(x.shape))))


def meanflat(x):
  return tf.reduce_mean(x, axis=list(range(1, len(x.shape))))


# ===== Neural network layers =====

def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return tf.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


def nin(x, *, name, num_units, init_scale=1.):
  with tf.variable_scope(name):
    in_dim = int(x.shape[-1])
    W = tf.get_variable('W', shape=[in_dim, num_units], initializer=default_init(scale=init_scale), dtype=DEFAULT_DTYPE)
    b = tf.get_variable('b', shape=[num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE)
    y = contract_inner(x, W) + b
    assert y.shape == x.shape[:-1] + [num_units]
    return y


def dense(x, *, name, num_units, init_scale=1., bias=True):
  with tf.variable_scope(name):
    _, in_dim = x.shape
    W = tf.get_variable('W', shape=[in_dim, num_units], initializer=default_init(scale=init_scale), dtype=DEFAULT_DTYPE)
    z = tf.matmul(x, W)
    if not bias:
      return z
    b = tf.get_variable('b', shape=[num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE)
    return z + b


def conv2d(x, *, name, num_units, filter_size=(3, 3), stride=1, dilation=None, pad='SAME', init_scale=1., bias=True):
  with tf.variable_scope(name):
    assert x.shape.ndims == 4
    if isinstance(filter_size, int):
      filter_size = (filter_size, filter_size)
    W = tf.get_variable('W', shape=[*filter_size, int(x.shape[-1]), num_units],
                        initializer=default_init(scale=init_scale), dtype=DEFAULT_DTYPE)
    z = tf.nn.conv2d(x, W, strides=stride, padding=pad, dilations=dilation)
    if not bias:
      return z
    b = tf.get_variable('b', shape=[num_units], initializer=tf.constant_initializer(0.), dtype=DEFAULT_DTYPE)
    return z + b


def get_timestep_embedding(timesteps, embedding_dim: int):
  """
  From Fairseq.
  Build sinusoidal embeddings.
  This matches the implementation in tensor2tensor, but differs slightly
  from the description in Section 3.5 of "Attention Is All You Need".
  """
  assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32

  half_dim = embedding_dim // 2
  emb = math.log(10000) / (half_dim - 1)
  emb = tf.exp(tf.range(half_dim, dtype=DEFAULT_DTYPE) * -emb)
  # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = tf.cast(timesteps, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
  emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=1)
  if embedding_dim % 2 == 1:  # zero pad
    # emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
    emb = tf.pad(emb, [[0, 0], [0, 1]])
  assert emb.shape == [timesteps.shape[0], embedding_dim]
  return emb
