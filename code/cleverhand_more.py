from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from cleverhans.compat import reduce_mean, reduce_prod
from cleverhans.model import Model
from cleverhans.serial import PicklableVariable as PV
from cleverhans.utils import ordered_union
from cleverhans.picklable_model import *

class InstanceNorm(Layer):

  def __init__(self, eps=1e-3, init_gamma=1.,
               **kwargs):
    self.eps = eps
    self.init_gamma = init_gamma
    super(InstanceNorm, self).__init__(**kwargs)

  def set_input_shape(self, shape):
    self.input_shape = shape
    self.output_shape = shape
    channels = shape[-1]
    init_value = np.ones((channels,), dtype='float32') * self.init_gamma
    self.gamma = PV(init_value, name=self.name + "_gamma")
    self.beta = PV(np.zeros((channels,), dtype='float32'),
                   name=self.name + "_beta")

  def fprop(self, x, **kwargs):
    mean, var = tf.nn.moments(x, [0, 1], keep_dims=True)
    x = (x - mean) * tf.rsqrt(var + self.eps)
    x = x * self.gamma.var + self.beta.var
    return x

  def get_params(self):
    return [self.gamma.var, self.beta.var]


class ResidualWithInstanceNorm(Layer):
  """A residual network layer that uses batch normalization.
  :param out_filter: Number of output filters
  :param stride: int
      Stride for convolutional layers. Replicated to both row and column.
  """

  def __init__(self, out_filter, stride, activate_before_residual=False,
               leak=0.1, **kwargs):
    assert isinstance(stride, int)
    self.__dict__.update(locals())
    del self.self
    self.lrelu = LeakyReLU(leak)
    super(ResidualWithInstanceNorm, self).__init__(**kwargs)

  def set_input_shape(self, shape):
    self.input_shape = tuple(shape)
    self.in_filter = shape[-1]
    self.bn1 = InstanceNorm(name=self.name + "_bn1")
    self.bn1.set_input_shape(shape)
    strides = (self.stride, self.stride)
    self.conv1 = Conv2D(self.out_filter, (3, 3), strides, "SAME",
                        name=self.name + "_conv1", init_mode="inv_sqrt")
    self.conv1.set_input_shape(shape)
    self.bn2 = InstanceNorm(name=self.name + "_bn2")
    self.bn2.set_input_shape(self.conv1.get_output_shape())
    self.conv2 = Conv2D(self.out_filter, (3, 3), (1, 1), "SAME",
                        name=self.name + "_conv2", init_mode="inv_sqrt")
    self.conv2.set_input_shape(self.conv1.get_output_shape())
    self.output_shape = self.conv2.get_output_shape()

  def get_params(self):
    sublayers = [self.conv1, self.conv2, self.bn1, self.bn2]
    params = []
    for sublayer in sublayers:
      params = params + sublayer.get_params()
    assert self.conv1.kernels.var in params
    return params

  def fprop(self, x, **kwargs):
    if self.activate_before_residual:
      x = self.bn1.fprop(x)
      x = self.lrelu.fprop(x)
      orig_x = x
    else:
      orig_x = x
      x = self.bn1.fprop(x)
      x = self.lrelu.fprop(x)
    x = self.conv1.fprop(x)
    x = self.bn2.fprop(x)
    x = self.lrelu.fprop(x)
    x = self.conv2.fprop(x)
    if self.stride != 1:
      stride = [1, self.stride, self.stride, 1]
      orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
    out_filter = self.out_filter
    in_filter = self.in_filter
    if in_filter != out_filter:
      orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0],
                               [(out_filter - in_filter) // 2,
                                (out_filter - in_filter) // 2]])
    x = x + orig_x
    return x
