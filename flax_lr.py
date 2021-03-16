# Copyright 2021 The Flax Authors.
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

import jax
from jax import numpy as jnp, random, lax
import flax
from flax import linen
from flax.nn import initializers
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, Union
from flax.linen import Module, compact, Dense
#import numpy as np
#from pprint import pprint
#from dense import Dense

from jax.config import config
jax.config.update('jax_enable_x64', True)

# Require JAX omnistaging mode.
jax.config.enable_omnistaging()



class LR(Module):
  sizes: Iterable[int]

  @compact
  def __call__(self, x):
    return Dense(self.sizes[0])(x)
