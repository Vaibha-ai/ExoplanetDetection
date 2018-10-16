# Copyright 2018 The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generates synthetic light curves with periodic transit-like dips.

See class docstring below for more information.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class SyntheticTransitMaker(object):
  """Generates synthetic light curves with periodic transit-like dips.

  These light curves are generated by thresholding noisy sine waves. Each time
  random_light_curve is called, a thresholded sine wave is generated by sampling
  parameters uniformly from the ranges specified below.

  Attributes:
    period_range: A tuple of positive values specifying the range of periods the
      sine waves may take.
    amplitude_range: A tuple of positive values specifying the range of
      amplitudes the sine waves may take.
    threshold_ratio_range: A tuple of values in [0, 1) specifying the range of
      thresholds as a ratio of the sine wave amplitude.
    phase_range: Tuple of values specifying the range of phases the sine wave
      may take as a ratio of the sampled period. E.g. a sampled phase of 0.5
      would translate the sine wave by half of the period. The most common
      reason to override this would be to generate light curves
      deterministically (with e.g. (0,0)).
    noise_sd_range: A tuple of values in [0, 1) specifying the range of standard
      deviations for the Gaussian noise applied to the sine wave.
  """

  def __init__(self,
               period_range=(0.5, 4),
               amplitude_range=(1, 1),
               threshold_ratio_range=(0, 0.99),
               phase_range=(0, 1),
               noise_sd_range=(0.1, 0.1)):

    if threshold_ratio_range[0] < 0 or threshold_ratio_range[1] >= 1:
      raise ValueError("Threshold ratio range must be in [0, 1). Got: {}."
                       .format(threshold_ratio_range))
    if amplitude_range[0] <= 0:
      raise ValueError(
          "Amplitude range must only contain positive numbers. Got: {}.".format(
              amplitude_range))
    if period_range[0] <= 0:
      raise ValueError(
          "Period range must only contain positive numbers. Got: {}.".format(
              period_range))
    if noise_sd_range[0] < 0:
      raise ValueError(
          "Noise standard deviation range must be nonnegative. Got: {}.".format(
              noise_sd_range))

    for (start, end), name in [(period_range, "period"),
                               (amplitude_range, "amplitude"),
                               (threshold_ratio_range, "threshold ratio"),
                               (phase_range, "phase range"),
                               (noise_sd_range, "noise standard deviation")]:
      if end < start:
        raise ValueError(
            "End of {} range may not be less than start. Got: ({}, {})".format(
                name, start, end))

    self.period_range = period_range
    self.amplitude_range = amplitude_range
    self.threshold_ratio_range = threshold_ratio_range
    self.phase_range = phase_range
    self.noise_sd_range = noise_sd_range

  def random_light_curve(self, time, mask_prob=0):
    """Samples parameters and generates a light curve.

    Args:
      time: np.array, x-values to sample from the thresholded sine wave.
      mask_prob: value in [0,1], probability an individual datapoint is set to
        zero

    Returns:
      flux: np.array, values of the masked sampled light curve corresponding to
        the provided time array.
      mask: np.array of ones and zeros, with zeros indicating masking at the
        respective position on the flux array.
    """

    period = np.random.uniform(*self.period_range)
    phase = np.random.uniform(*self.phase_range) * period
    amplitude = np.random.uniform(*self.amplitude_range)
    threshold = np.random.uniform(*self.threshold_ratio_range) * amplitude

    sin_wave = np.sin(time / period - phase) * amplitude
    flux = np.minimum(sin_wave, -threshold) + threshold

    noise_sd = np.random.uniform(*self.noise_sd_range)
    noise = np.random.normal(scale=noise_sd, size=(len(time),))
    flux += noise

    # Array of ones and zeros, where zeros indicate masking.
    mask = np.random.random(len(time)) > mask_prob
    mask = mask.astype(np.float)

    return flux * mask, mask

  def random_light_curve_generator(self, time, mask_prob=0):
    """Returns a generator function yielding random light curves.

    Args:
       time: An np.array of x-values to sample from the thresholded sine wave.
       mask_prob: Value in [0,1], probability an individual datapoint is set to
         zero.

    Returns:
      A generator yielding random light curves.
    """

    def generator_fn():
      while True:
        yield self.random_light_curve(time, mask_prob)

    return generator_fn
