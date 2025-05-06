# This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.

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

"""Text-centric preprocessing ops.

All preprocessing ops should return a data processing functors. A data
is represented as a dictionary of (TF) tensors. The functors output a modified
dictionary.

A commonly used key for the tokenized output is "labels".
"""
from src.datasets import in1k_class_names as imagenet_class_names
from src.helpers.registry import InKeyOutKey, Registry

import tensorflow as tf


@Registry.register("preprocess_ops.clip_i1k_label_names")
@InKeyOutKey(indefault="label", outdefault="labels")
def get_pp_clip_i1k_label_names():
  """Convert i1k label numbers to strings, using CLIP's class names."""

  def _pp_imagenet_labels(label):
    return tf.gather(imagenet_class_names.CLIP_IMAGENET_CLASS_NAMES, label)

  return _pp_imagenet_labels

@Registry.register("preprocess_ops.get_autoreg_label")
@InKeyOutKey(indefault="labels_for_regress", outdefault="autoreg_labels")
def get_pp_autoreg_label(pad_token):
  """Convert i1k label numbers to strings, using CLIP's class names."""

  def _pp_autoreg_label(label):
    
    shift_label = label[1:]  # drop the <bos> token
    
    shift_label = tf.concat([shift_label, tf.constant([pad_token,])], axis=0)

    return shift_label

  return _pp_autoreg_label