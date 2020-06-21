from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from absl import logging
import tensorflow as tf


summary = tf.contrib.summary  # TensorFlow Summary API v2.


TpuSummaryEntry = collections.namedtuple(
    "TpuSummaryEntry", "summary_fn name tensor reduce_fn")


class TpuSummaries(object):
  """Class to simplify TF summaries on TPU.

  An instance of the class provides simple methods for writing summaries in the
  similar way to tf.summary. The difference is that each summary entry must
  provide a reduction function that is used to reduce the summary values from
  all the TPU cores.
  """

  def __init__(self, log_dir, save_summary_steps=250):
    self._log_dir = log_dir
    self._entries = []
    # While False no summary entries will be added. On TPU we unroll the graph
    # and don't want to add multiple summaries per step.
    self.record = True
    self._save_summary_steps = save_summary_steps

  def image(self, name, tensor, reduce_fn):
    """Add a summary for images. Tensor must be of 4-D tensor."""
    if not self.record:
      return
    self._entries.append(
        TpuSummaryEntry(summary.image, name, tensor, reduce_fn))

  def scalar(self, name, tensor, reduce_fn=tf.math.reduce_mean):
    """Add a summary for a scalar tensor."""
    if not self.record:
      return
    tensor = tf.convert_to_tensor(tensor)
    if tensor.shape.ndims == 0:
      tensor = tf.expand_dims(tensor, 0)
    self._entries.append(
        TpuSummaryEntry(summary.scalar, name, tensor, reduce_fn))

  def get_host_call(self):
    """Returns the tuple (host_call_fn, host_call_args) for TPUEstimatorSpec."""
    # All host_call_args must be tensors with batch dimension.
    # All tensors are streamed to the host machine (mind the band width).
    global_step = tf.train.get_or_create_global_step()
    host_call_args = [tf.expand_dims(global_step, 0)]
    host_call_args.extend([e.tensor for e in self._entries])
    logging.info("host_call_args: %s", host_call_args)
    return (self._host_call_fn, host_call_args)

  def _host_call_fn(self, step, *args):
    """Function that will run on the host machine."""
    # Host call receives values from all tensor cores (concatenate on the
    # batch dimension). Step is the same for all cores.
    step = step[0]
    logging.info("host_call_fn: args=%s", args)
    with summary.create_file_writer(self._log_dir).as_default():
      with summary.record_summaries_every_n_global_steps(
          self._save_summary_steps, step):
        for i, e in enumerate(self._entries):
          value = e.reduce_fn(args[i])
          e.summary_fn(e.name, value, step=step)
        return summary.all_summary_ops()
