"""
Direct NumPy port of tfgan.eval.classifier_metrics
"""

import numpy as np
import scipy.special


def log_softmax(x, axis):
  return x - scipy.special.logsumexp(x, axis=axis, keepdims=True)


def kl_divergence(p, p_logits, q):
  assert len(p.shape) == len(p_logits.shape) == 2
  assert len(q.shape) == 1
  return np.sum(p * (log_softmax(p_logits, axis=1) - np.log(q)[None, :]), axis=1)


def _symmetric_matrix_square_root(mat, eps=1e-10):
  """Compute square root of a symmetric matrix.

  Note that this is different from an elementwise square root. We want to
  compute M' where M' = sqrt(mat) such that M' * M' = mat.

  Also note that this method **only** works for symmetric matrices.

  Args:
    mat: Matrix to take the square root of.
    eps: Small epsilon such that any element less than eps will not be square
      rooted to guard against numerical instability.

  Returns:
    Matrix square root of mat.
  """
  u, s, vt = np.linalg.svd(mat)
  # sqrt is unstable around 0, just use 0 in such case
  si = np.where(s < eps, s, np.sqrt(s))
  return u.dot(np.diag(si)).dot(vt)


def trace_sqrt_product(sigma, sigma_v):
  """Find the trace of the positive sqrt of product of covariance matrices.

  '_symmetric_matrix_square_root' only works for symmetric matrices, so we
  cannot just take _symmetric_matrix_square_root(sigma * sigma_v).
  ('sigma' and 'sigma_v' are symmetric, but their product is not necessarily).

  Let sigma = A A so A = sqrt(sigma), and sigma_v = B B.
  We want to find trace(sqrt(sigma sigma_v)) = trace(sqrt(A A B B))
  Note the following properties:
  (i) forall M1, M2: eigenvalues(M1 M2) = eigenvalues(M2 M1)
     => eigenvalues(A A B B) = eigenvalues (A B B A)
  (ii) if M1 = sqrt(M2), then eigenvalues(M1) = sqrt(eigenvalues(M2))
     => eigenvalues(sqrt(sigma sigma_v)) = sqrt(eigenvalues(A B B A))
  (iii) forall M: trace(M) = sum(eigenvalues(M))
     => trace(sqrt(sigma sigma_v)) = sum(eigenvalues(sqrt(sigma sigma_v)))
                                   = sum(sqrt(eigenvalues(A B B A)))
                                   = sum(eigenvalues(sqrt(A B B A)))
                                   = trace(sqrt(A B B A))
                                   = trace(sqrt(A sigma_v A))
  A = sqrt(sigma). Both sigma and A sigma_v A are symmetric, so we **can**
  use the _symmetric_matrix_square_root function to find the roots of these
  matrices.

  Args:
    sigma: a square, symmetric, real, positive semi-definite covariance matrix
    sigma_v: same as sigma

  Returns:
    The trace of the positive square root of sigma*sigma_v
  """

  # Note sqrt_sigma is called "A" in the proof above
  sqrt_sigma = _symmetric_matrix_square_root(sigma)

  # This is sqrt(A sigma_v A) above
  sqrt_a_sigmav_a = sqrt_sigma.dot(sigma_v.dot(sqrt_sigma))

  return np.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def classifier_score_from_logits(logits):
  """Classifier score for evaluating a generative model from logits.

  This method computes the classifier score for a set of logits. This can be
  used independently of the classifier_score() method, especially in the case
  of using large batches during evaluation where we would like precompute all
  of the logits before computing the classifier score.

  This technique is described in detail in https://arxiv.org/abs/1606.03498. In
  summary, this function calculates:

  exp( E[ KL(p(y|x) || p(y)) ] )

  which captures how different the network's classification prediction is from
  the prior distribution over classes.

  Args:
    logits: Precomputed 2D tensor of logits that will be used to compute the
      classifier score.

  Returns:
    The classifier score. A floating-point scalar of the same type as the output
    of `logits`.
  """
  assert len(logits.shape) == 2

  # Use maximum precision for best results.
  logits_dtype = logits.dtype
  if logits_dtype != np.float64:
    logits = logits.astype(np.float64)

  p = scipy.special.softmax(logits, axis=1)
  q = np.mean(p, axis=0)
  kl = kl_divergence(p, logits, q)
  assert len(kl.shape) == 1
  log_score = np.mean(kl)
  final_score = np.exp(log_score)

  if logits_dtype != np.float64:
    final_score = final_score.astype(logits_dtype)

  return final_score


def frechet_classifier_distance_from_activations(real_activations,
                                                 generated_activations):
  """Classifier distance for evaluating a generative model.

  This methods computes the Frechet classifier distance from activations of
  real images and generated images. This can be used independently of the
  frechet_classifier_distance() method, especially in the case of using large
  batches during evaluation where we would like precompute all of the
  activations before computing the classifier distance.

  This technique is described in detail in https://arxiv.org/abs/1706.08500.
  Given two Gaussian distribution with means m and m_w and covariance matrices
  C and C_w, this function calculates

                |m - m_w|^2 + Tr(C + C_w - 2(C * C_w)^(1/2))

  which captures how different the distributions of real images and generated
  images (or more accurately, their visual features) are. Note that unlike the
  Inception score, this is a true distance and utilizes information about real
  world images.

  Note that when computed using sample means and sample covariance matrices,
  Frechet distance is biased. It is more biased for small sample sizes. (e.g.
  even if the two distributions are the same, for a small sample size, the
  expected Frechet distance is large). It is important to use the same
  sample size to compute frechet classifier distance when comparing two
  generative models.

  Args:
    real_activations: 2D Tensor containing activations of real data. Shape is
      [batch_size, activation_size].
    generated_activations: 2D Tensor containing activations of generated data.
      Shape is [batch_size, activation_size].

  Returns:
   The Frechet Inception distance. A floating-point scalar of the same type
   as the output of the activations.

  """
  assert len(real_activations.shape) == len(generated_activations.shape) == 2

  activations_dtype = real_activations.dtype
  if activations_dtype != np.float64:
    real_activations = real_activations.astype(np.float64)
    generated_activations = generated_activations.astype(np.float64)

  # Compute mean and covariance matrices of activations.
  m = np.mean(real_activations, 0)
  m_w = np.mean(generated_activations, 0)
  num_examples_real = float(real_activations.shape[0])
  num_examples_generated = float(generated_activations.shape[0])

  # sigma = (1 / (n - 1)) * (X - mu) (X - mu)^T
  real_centered = real_activations - m
  sigma = real_centered.T.dot(real_centered) / (num_examples_real - 1)

  gen_centered = generated_activations - m_w
  sigma_w = gen_centered.T.dot(gen_centered) / (num_examples_generated - 1)

  # Find the Tr(sqrt(sigma sigma_w)) component of FID
  sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)

  # Compute the two components of FID.

  # First the covariance component.
  # Here, note that trace(A + B) = trace(A) + trace(B)
  trace = np.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component

  # Next the distance between means.
  mean = np.sum(np.square(m - m_w))  # Equivalent to L2 but more stable.
  fid = trace + mean
  if activations_dtype != np.float64:
    fid = fid.astype(activations_dtype)

  return fid


def test_all():
  """
  Test against tfgan.eval.classifier_metrics
  """

  import tensorflow.compat.v1 as tf
  import tensorflow_gan as tfgan

  rand = np.random.RandomState(1234)
  logits = rand.randn(64, 1008)
  asdf1, asdf2 = rand.randn(64, 2048), rand.rand(256, 2048)
  with tf.Session() as sess:
    assert np.allclose(
      sess.run(tfgan.eval.classifier_score_from_logits(tf.convert_to_tensor(logits))),
      classifier_score_from_logits(logits))
    assert np.allclose(
      sess.run(tfgan.eval.frechet_classifier_distance_from_activations(
        tf.convert_to_tensor(asdf1), tf.convert_to_tensor(asdf2))),
      frechet_classifier_distance_from_activations(asdf1, asdf2))
  print('all ok')


if __name__ == '__main__':
  test_all()
