# This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.
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
from jax.experimental import mesh_utils
import logging
import numpy as np
from src.helpers.utils import tree_broadcast
import flax


def unbox_logicallypartioned(boxed_pytree):
  """Unboxes the flax.LogicallyPartitioned pieces

  Args:
    boxed_pytree: a pytree that includes LogicallyPartitioned
      leaves.
  Returns:
    a pytree where all all LogicallyPartitioned leaves have been unboxed.
  """
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax.linen.spmd.LogicallyPartitioned) else x,
      boxed_pytree,
      is_leaf=lambda k: isinstance(k, flax.linen.spmd.LogicallyPartitioned),
  )


def create_mesh(config):
    """Creates a device mesh with each slice in its own data parallel group."""
    #use for determine the shape of devices
    parallelism = [
        config.sharding.meshshape.data_parallelism,
        config.sharding.meshshape.fsdp_parallelism,
        config.sharding.meshshape.tensor_parallelism,
    ]

    mesh = mesh_utils.create_device_mesh(parallelism)

    logging.info("\u001b[33mNOTE\u001b[0m: " + f"Num_devices: {len(jax.devices())}, shape {mesh.shape}")

    return mesh




def reshard(tree, shardings):
    """Take an arbitrarily* sharded pytree and shard it according to `shardings`.

    This is a no-op for tree elements which are already sharded as requested.

    *Arrays that are fully addressable (for example, CPU arrays) are assumed to be
    identical (i.e. replicated) across hosts.

    *It does not work if an element of `tree` is not fully-addressable, unless its
    sharding is already consistent with the target sharding.
    If this is needed, please ping lbeyer@ or akolesnikov@.

    Args:
      tree: a pytree of arrays.
      shardings: a (prefix) pytree of jax array shardings.
    Returns:
      A pytree of global jax arrays that follows provided shardings.
    """
    def _make_global_arr(x, shard, shape):
        # Avoid unnecessary copies and transfers:
        if hasattr(x, "sharding") and x.sharding.is_equivalent_to(
                shard, len(shape)):  # pylint: disable=line-too-long
            return x
        if not getattr(x, "is_fully_addressable", True):
            raise RuntimeError(
                "Trying to reshard a non-fully-addressable array. "
                "Please see the doc-comment for detailed explanation.")
        x = jax.device_get(x)  # Might be on local devices.
        xs = [jax.device_put(x[s], device=d)
              for d, s in shard.addressable_devices_indices_map(shape).items()]
        return jax.make_array_from_single_device_arrays(shape, shard, xs)

    shapes = jax.tree_map(np.shape, tree)
    shardings = tree_broadcast(shardings, tree)
    return jax.tree_map(_make_global_arr, tree, shardings, shapes)


