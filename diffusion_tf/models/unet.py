import tensorflow.compat.v1 as tf
import tensorflow.contrib as tf_contrib

from .. import nn


def nonlinearity(x):
  return tf.nn.swish(x)


def normalize(x, *, temb, name):
  return tf_contrib.layers.group_norm(x, scope=name)


def upsample(x, *, name, with_conv):
  with tf.variable_scope(name):
    B, H, W, C = x.shape
    x = tf.image.resize(x, size=[H * 2, W * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)
    assert x.shape == [B, H * 2, W * 2, C]
    if with_conv:
      x = nn.conv2d(x, name='conv', num_units=C, filter_size=3, stride=1)
      assert x.shape == [B, H * 2, W * 2, C]
    return x


def downsample(x, *, name, with_conv):
  with tf.variable_scope(name):
    B, H, W, C = x.shape
    if with_conv:
      x = nn.conv2d(x, name='conv', num_units=C, filter_size=3, stride=2)
    else:
      x = tf.nn.avg_pool(x, 2, 2, 'SAME')
    assert x.shape == [B, H // 2, W // 2, C]
    return x


def resnet_block(x, *, temb, name, out_ch=None, conv_shortcut=False, dropout):
  B, H, W, C = x.shape
  if out_ch is None:
    out_ch = C

  with tf.variable_scope(name):
    h = x

    h = nonlinearity(normalize(h, temb=temb, name='norm1'))
    h = nn.conv2d(h, name='conv1', num_units=out_ch)

    # add in timestep embedding
    h += nn.dense(nonlinearity(temb), name='temb_proj', num_units=out_ch)[:, None, None, :]

    h = nonlinearity(normalize(h, temb=temb, name='norm2'))
    h = tf.nn.dropout(h, rate=dropout)
    h = nn.conv2d(h, name='conv2', num_units=out_ch, init_scale=0.)

    if C != out_ch:
      if conv_shortcut:
        x = nn.conv2d(x, name='conv_shortcut', num_units=out_ch)
      else:
        x = nn.nin(x, name='nin_shortcut', num_units=out_ch)

    assert x.shape == h.shape
    print('{}: x={} temb={}'.format(tf.get_default_graph().get_name_scope(), x.shape, temb.shape))
    return x + h


def attn_block(x, *, name, temb):
  B, H, W, C = x.shape
  with tf.variable_scope(name):
    h = normalize(x, temb=temb, name='norm')
    q = nn.nin(h, name='q', num_units=C)
    k = nn.nin(h, name='k', num_units=C)
    v = nn.nin(h, name='v', num_units=C)

    w = tf.einsum('bhwc,bHWc->bhwHW', q, k) * (int(C) ** (-0.5))
    w = tf.reshape(w, [B, H, W, H * W])
    w = tf.nn.softmax(w, -1)
    w = tf.reshape(w, [B, H, W, H, W])

    h = tf.einsum('bhwHW,bHWc->bhwc', w, v)
    h = nn.nin(h, name='proj_out', num_units=C, init_scale=0.)

    assert h.shape == x.shape
    print(tf.get_default_graph().get_name_scope(), x.shape)
    return x + h


def model(x, *, t, y, name, num_classes, reuse=tf.AUTO_REUSE, ch, out_ch, ch_mult=(1, 2, 4, 8), num_res_blocks,
          attn_resolutions, dropout=0., resamp_with_conv=True):
  B, S, _, _ = x.shape
  assert x.dtype == tf.float32 and x.shape[2] == S
  assert t.dtype in [tf.int32, tf.int64]
  num_resolutions = len(ch_mult)

  assert num_classes == 1 and y is None, 'not supported'
  del y

  with tf.variable_scope(name, reuse=reuse):
    # Timestep embedding
    with tf.variable_scope('temb'):
      temb = nn.get_timestep_embedding(t, ch)
      temb = nn.dense(temb, name='dense0', num_units=ch * 4)
      temb = nn.dense(nonlinearity(temb), name='dense1', num_units=ch * 4)
      assert temb.shape == [B, ch * 4]

    # Downsampling
    hs = [nn.conv2d(x, name='conv_in', num_units=ch)]
    for i_level in range(num_resolutions):
      with tf.variable_scope('down_{}'.format(i_level)):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks):
          h = resnet_block(
            hs[-1], name='block_{}'.format(i_block), temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
          if h.shape[1] in attn_resolutions:
            h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
          hs.append(h)
        # Downsample
        if i_level != num_resolutions - 1:
          hs.append(downsample(hs[-1], name='downsample', with_conv=resamp_with_conv))

    # Middle
    with tf.variable_scope('mid'):
      h = hs[-1]
      h = resnet_block(h, temb=temb, name='block_1', dropout=dropout)
      h = attn_block(h, name='attn_1'.format(i_block), temb=temb)
      h = resnet_block(h, temb=temb, name='block_2', dropout=dropout)

    # Upsampling
    for i_level in reversed(range(num_resolutions)):
      with tf.variable_scope('up_{}'.format(i_level)):
        # Residual blocks for this resolution
        for i_block in range(num_res_blocks + 1):
          h = resnet_block(tf.concat([h, hs.pop()], axis=-1), name='block_{}'.format(i_block),
                           temb=temb, out_ch=ch * ch_mult[i_level], dropout=dropout)
          if h.shape[1] in attn_resolutions:
            h = attn_block(h, name='attn_{}'.format(i_block), temb=temb)
        # Upsample
        if i_level != 0:
          h = upsample(h, name='upsample', with_conv=resamp_with_conv)
    assert not hs

    # End
    h = nonlinearity(normalize(h, temb=temb, name='norm_out'))
    h = nn.conv2d(h, name='conv_out', num_units=out_ch, init_scale=0.)
    assert h.shape == x.shape[:3] + [out_ch]
    return h
