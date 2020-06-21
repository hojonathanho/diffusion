import numpy as np
import tensorflow.compat.v1 as tf

from . import nn
from . import utils


def normal_kl(mean1, logvar1, mean2, logvar2):
  """
  KL divergence between normal distributions parameterized by mean and log-variance.
  """
  return 0.5 * (-1.0 + logvar2 - logvar1 + tf.exp(logvar1 - logvar2)
                + tf.squared_difference(mean1, mean2) * tf.exp(-logvar2))


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
  betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  warmup_time = int(num_diffusion_timesteps * warmup_frac)
  betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
  return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
  if beta_schedule == 'quad':
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
  elif beta_schedule == 'linear':
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'warmup10':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
  elif beta_schedule == 'warmup50':
    betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
  elif beta_schedule == 'const':
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
  elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
    betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
  else:
    raise NotImplementedError(beta_schedule)
  assert betas.shape == (num_diffusion_timesteps,)
  return betas


class GaussianDiffusion2:
  """
  Contains utilities for the diffusion model.

  Arguments:
  - what the network predicts (x_{t-1}, x_0, or epsilon)
  - which loss function (kl or unweighted MSE)
  - what is the variance of p(x_{t-1}|x_t) (learned, fixed to beta, or fixed to weighted beta)
  - what type of decoder, and how to weight its loss? is its variance learned too?
  """

  def __init__(self, *, betas, model_mean_type, model_var_type, loss_type):
    self.model_mean_type = model_mean_type  # xprev, xstart, eps
    self.model_var_type = model_var_type  # learned, fixedsmall, fixedlarge
    self.loss_type = loss_type  # kl, mse

    assert isinstance(betas, np.ndarray)
    self.betas = betas = betas.astype(np.float64)  # computations here in float64 for accuracy
    assert (betas > 0).all() and (betas <= 1).all()
    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)

    alphas = 1. - betas
    self.alphas_cumprod = np.cumprod(alphas, axis=0)
    self.alphas_cumprod_prev = np.append(1., self.alphas_cumprod[:-1])
    assert self.alphas_cumprod_prev.shape == (timesteps,)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = np.sqrt(1. - self.alphas_cumprod)
    self.log_one_minus_alphas_cumprod = np.log(1. - self.alphas_cumprod)
    self.sqrt_recip_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = np.sqrt(1. / self.alphas_cumprod - 1)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self.posterior_log_variance_clipped = np.log(np.append(self.posterior_variance[1], self.posterior_variance[1:]))
    self.posterior_mean_coef1 = betas * np.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1. - self.alphas_cumprod)

  @staticmethod
  def _extract(a, t, x_shape):
    """
    Extract some coefficients at specified timesteps,
    then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    bs, = t.shape
    assert x_shape[0] == bs
    out = tf.gather(tf.convert_to_tensor(a, dtype=tf.float32), t)
    assert out.shape == [bs]
    return tf.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

  def q_mean_variance(self, x_start, t):
    mean = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
    variance = self._extract(1. - self.alphas_cumprod, t, x_start.shape)
    log_variance = self._extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
    return mean, variance, log_variance

  def q_sample(self, x_start, t, noise=None):
    """
    Diffuse the data (t == 0 means diffused for 1 step)
    """
    if noise is None:
      noise = tf.random_normal(shape=x_start.shape)
    assert noise.shape == x_start.shape
    return (
        self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )

  def q_posterior_mean_variance(self, x_start, x_t, t):
    """
    Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
    """
    assert x_start.shape == x_t.shape
    posterior_mean = (
        self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
        self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
    assert (posterior_mean.shape[0] == posterior_variance.shape[0] == posterior_log_variance_clipped.shape[0] ==
            x_start.shape[0])
    return posterior_mean, posterior_variance, posterior_log_variance_clipped

  def p_mean_variance(self, denoise_fn, *, x, t, clip_denoised: bool, return_pred_xstart: bool):
    B, H, W, C = x.shape
    assert t.shape == [B]
    model_output = denoise_fn(x, t)

    # Learned or fixed variance?
    if self.model_var_type == 'learned':
      assert model_output.shape == [B, H, W, C * 2]
      model_output, model_log_variance = tf.split(model_output, 2, axis=-1)
      model_variance = tf.exp(model_log_variance)
    elif self.model_var_type in ['fixedsmall', 'fixedlarge']:
      # below: only log_variance is used in the KL computations
      model_variance, model_log_variance = {
        # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
        'fixedlarge': (self.betas, np.log(np.append(self.posterior_variance[1], self.betas[1:]))),
        'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
      }[self.model_var_type]
      model_variance = self._extract(model_variance, t, x.shape) * tf.ones(x.shape.as_list())
      model_log_variance = self._extract(model_log_variance, t, x.shape) * tf.ones(x.shape.as_list())
    else:
      raise NotImplementedError(self.model_var_type)

    # Mean parameterization
    _maybe_clip = lambda x_: (tf.clip_by_value(x_, -1., 1.) if clip_denoised else x_)
    if self.model_mean_type == 'xprev':  # the model predicts x_{t-1}
      pred_xstart = _maybe_clip(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
      model_mean = model_output
    elif self.model_mean_type == 'xstart':  # the model predicts x_0
      pred_xstart = _maybe_clip(model_output)
      model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
    elif self.model_mean_type == 'eps':  # the model predicts epsilon
      pred_xstart = _maybe_clip(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
      model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
    else:
      raise NotImplementedError(self.model_mean_type)

    assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
    if return_pred_xstart:
      return model_mean, model_variance, model_log_variance, pred_xstart
    else:
      return model_mean, model_variance, model_log_variance

  def _predict_xstart_from_eps(self, x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
        self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
    )

  def _predict_xstart_from_xprev(self, x_t, t, xprev):
    assert x_t.shape == xprev.shape
    return (  # (xprev - coef2*x_t) / coef1
        self._extract(1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
        self._extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
    )

  # === Sampling ===

  def p_sample(self, denoise_fn, *, x, t, noise_fn, clip_denoised=True, return_pred_xstart: bool):
    """
    Sample from the model
    """
    model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
      denoise_fn, x=x, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
    noise = noise_fn(shape=x.shape, dtype=x.dtype)
    assert noise.shape == x.shape
    # no noise when t == 0
    nonzero_mask = tf.reshape(1 - tf.cast(tf.equal(t, 0), tf.float32), [x.shape[0]] + [1] * (len(x.shape) - 1))
    sample = model_mean + nonzero_mask * tf.exp(0.5 * model_log_variance) * noise
    assert sample.shape == pred_xstart.shape
    return (sample, pred_xstart) if return_pred_xstart else sample

  def p_sample_loop(self, denoise_fn, *, shape, noise_fn=tf.random_normal):
    """
    Generate samples
    """
    assert isinstance(shape, (tuple, list))
    i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
    img_0 = noise_fn(shape=shape, dtype=tf.float32)
    _, img_final = tf.while_loop(
      cond=lambda i_, _: tf.greater_equal(i_, 0),
      body=lambda i_, img_: [
        i_ - 1,
        self.p_sample(
          denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), noise_fn=noise_fn, return_pred_xstart=False)
      ],
      loop_vars=[i_0, img_0],
      shape_invariants=[i_0.shape, img_0.shape],
      back_prop=False
    )
    assert img_final.shape == shape
    return img_final

  def p_sample_loop_progressive(self, denoise_fn, *, shape, noise_fn=tf.random_normal, include_xstartpred_freq=50):
    """
    Generate samples and keep track of prediction of x0
    """
    assert isinstance(shape, (tuple, list))
    i_0 = tf.constant(self.num_timesteps - 1, dtype=tf.int32)
    img_0 = noise_fn(shape=shape, dtype=tf.float32)  # [B, H, W, C]

    num_recorded_xstartpred = self.num_timesteps // include_xstartpred_freq
    xstartpreds_0 = tf.zeros([shape[0], num_recorded_xstartpred, *shape[1:]], dtype=tf.float32)  # [B, N, H, W, C]

    def _loop_body(i_, img_, xstartpreds_):
      # Sample p(x_{t-1} | x_t) as usual
      sample, pred_xstart = self.p_sample(
        denoise_fn=denoise_fn, x=img_, t=tf.fill([shape[0]], i_), noise_fn=noise_fn, return_pred_xstart=True)
      assert sample.shape == pred_xstart.shape == shape
      # Keep track of prediction of x0
      insert_mask = tf.equal(tf.floordiv(i_, include_xstartpred_freq),
                             tf.range(num_recorded_xstartpred, dtype=tf.int32))
      insert_mask = tf.reshape(tf.cast(insert_mask, dtype=tf.float32),
                               [1, num_recorded_xstartpred, *([1] * len(shape[1:]))])  # [1, N, 1, 1, 1]
      new_xstartpreds = insert_mask * pred_xstart[:, None, ...] + (1. - insert_mask) * xstartpreds_
      return [i_ - 1, sample, new_xstartpreds]

    _, img_final, xstartpreds_final = tf.while_loop(
      cond=lambda i_, img_, xstartpreds_: tf.greater_equal(i_, 0),
      body=_loop_body,
      loop_vars=[i_0, img_0, xstartpreds_0],
      shape_invariants=[i_0.shape, img_0.shape, xstartpreds_0.shape],
      back_prop=False
    )
    assert img_final.shape == shape and xstartpreds_final.shape == xstartpreds_0.shape
    return img_final, xstartpreds_final  # xstart predictions should agree with img_final at step 0

  # === Log likelihood calculation ===

  def _vb_terms_bpd(self, denoise_fn, x_start, x_t, t, *, clip_denoised: bool, return_pred_xstart: bool):
    true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
    model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
      denoise_fn, x=x_t, t=t, clip_denoised=clip_denoised, return_pred_xstart=True)
    kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
    kl = nn.meanflat(kl) / np.log(2.)

    decoder_nll = -utils.discretized_gaussian_log_likelihood(
      x_start, means=model_mean, log_scales=0.5 * model_log_variance)
    assert decoder_nll.shape == x_start.shape
    decoder_nll = nn.meanflat(decoder_nll) / np.log(2.)

    # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    assert kl.shape == decoder_nll.shape == t.shape == [x_start.shape[0]]
    output = tf.where(tf.equal(t, 0), decoder_nll, kl)
    return (output, pred_xstart) if return_pred_xstart else output

  def training_losses(self, denoise_fn, x_start, t, noise=None):
    """
    Training loss calculation
    """

    # Add noise to data
    assert t.shape == [x_start.shape[0]]
    if noise is None:
      noise = tf.random_normal(shape=x_start.shape, dtype=x_start.dtype)
    assert noise.shape == x_start.shape and noise.dtype == x_start.dtype
    x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

    # Calculate the loss
    if self.loss_type == 'kl':  # the variational bound
      losses = self._vb_terms_bpd(
        denoise_fn=denoise_fn, x_start=x_start, x_t=x_t, t=t, clip_denoised=False, return_pred_xstart=False)
    elif self.loss_type == 'mse':  # unweighted MSE
      assert self.model_var_type != 'learned'
      target = {
        'xprev': self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0],
        'xstart': x_start,
        'eps': noise
      }[self.model_mean_type]
      model_output = denoise_fn(x_t, t)
      assert model_output.shape == target.shape == x_start.shape
      losses = nn.meanflat(tf.squared_difference(target, model_output))
    else:
      raise NotImplementedError(self.loss_type)

    assert losses.shape == t.shape
    return losses

  def _prior_bpd(self, x_start):
    B, T = x_start.shape[0], self.num_timesteps
    qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t=tf.fill([B], tf.constant(T - 1, dtype=tf.int32)))
    kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0., logvar2=0.)
    assert kl_prior.shape == x_start.shape
    return nn.meanflat(kl_prior) / np.log(2.)

  def calc_bpd_loop(self, denoise_fn, x_start, *, clip_denoised=True):
    (B, H, W, C), T = x_start.shape, self.num_timesteps

    def _loop_body(t_, cur_vals_bt_, cur_mse_bt_):
      assert t_.shape == []
      t_b = tf.fill([B], t_)
      # Calculate VLB term at the current timestep
      new_vals_b, pred_xstart = self._vb_terms_bpd(
        denoise_fn, x_start=x_start, x_t=self.q_sample(x_start=x_start, t=t_b), t=t_b,
        clip_denoised=clip_denoised, return_pred_xstart=True)
      # MSE for progressive prediction loss
      assert pred_xstart.shape == x_start.shape
      new_mse_b = nn.meanflat(tf.squared_difference(pred_xstart, x_start))
      assert new_vals_b.shape == new_mse_b.shape == [B]
      # Insert the calculated term into the tensor of all terms
      mask_bt = tf.cast(tf.equal(t_b[:, None], tf.range(T)[None, :]), dtype=tf.float32)
      new_vals_bt = cur_vals_bt_ * (1. - mask_bt) + new_vals_b[:, None] * mask_bt
      new_mse_bt = cur_mse_bt_ * (1. - mask_bt) + new_mse_b[:, None] * mask_bt
      assert mask_bt.shape == cur_vals_bt_.shape == new_vals_bt.shape == [B, T]
      return t_ - 1, new_vals_bt, new_mse_bt

    t_0 = tf.constant(T - 1, dtype=tf.int32)
    terms_0 = tf.zeros([B, T])
    mse_0 = tf.zeros([B, T])
    _, terms_bpd_bt, mse_bt = tf.while_loop(  # Note that this can be implemented with tf.map_fn instead
      cond=lambda t_, cur_vals_bt_, cur_mse_bt_: tf.greater_equal(t_, 0),
      body=_loop_body,
      loop_vars=[t_0, terms_0, mse_0],
      shape_invariants=[t_0.shape, terms_0.shape, mse_0.shape],
      back_prop=False
    )
    prior_bpd_b = self._prior_bpd(x_start)
    total_bpd_b = tf.reduce_sum(terms_bpd_bt, axis=1) + prior_bpd_b
    assert terms_bpd_bt.shape == mse_bt.shape == [B, T] and total_bpd_b.shape == prior_bpd_b.shape == [B]
    return total_bpd_b, terms_bpd_bt, prior_bpd_b, mse_bt
