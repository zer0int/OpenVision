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

"""A refactored and simplified ViT.

However, the names of modules are made to match the old ones for easy loading.
"""

import sys
from typing import Any, Callable, Optional, Sequence, Tuple, Union


from absl import logging
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage
from flax.linen.partitioning import remat


from src.helpers import utils
from src.models import common
from src.models.bpt import blockwise_ffn
from src.models.common import DropPath

Dtype = Any  # this could be a real type?

class CustomMultiheadAttention(nn.Module):
    d_model: int
    context_dim: int
    n_head: int = 8

    def setup(self):
        self.d_head = self.d_model // self.n_head
        self.q_proj = nn.Dense(features=self.d_model, use_bias=False)
        self.k_proj = nn.Dense(features=self.d_model, use_bias=False)  # Ensure the same output dimension as q_proj
        self.v_proj = nn.Dense(features=self.d_model, use_bias=False)  # Ensure the same output dimension as q_proj
        self.out_proj = nn.Dense(features=self.d_model, use_bias=False)

    def __call__(self, q, k, v):
        # Project inputs to q, k, v
        q_proj = self.q_proj(q)  # (N, L_q, d_model)
        k_proj = self.k_proj(k)  # (N, L_kv, d_model)
        v_proj = self.v_proj(v)  # (N, L_kv, d_model)


        # Reshape for multi-head attention
        q_proj = q_proj.reshape(q_proj.shape[0], q_proj.shape[1], self.n_head, self.d_head)
        k_proj = k_proj.reshape(k_proj.shape[0], k_proj.shape[1], self.n_head, self.d_head)
        v_proj = v_proj.reshape(v_proj.shape[0], v_proj.shape[1], self.n_head, self.d_head)

        # Transpose to (N, n_head, L_q/L_kv, d_head)
        q_proj = jnp.transpose(q_proj, (0, 2, 1, 3))  # (N, n_head, L_q, d_head)
        k_proj = jnp.transpose(k_proj, (0, 2, 1, 3))  # (N, n_head, L_kv, d_head)
        v_proj = jnp.transpose(v_proj, (0, 2, 1, 3))  # (N, n_head, L_kv, d_head)


        # Scaled dot-product attention
        attn_weights = jax.nn.softmax(jnp.einsum('nhqd,nhkd->nhqk', q_proj, k_proj) / jnp.sqrt(self.d_head), axis=-1)
        attn_output = jnp.einsum('nhqk,nhvd->nhqd', attn_weights, v_proj)


        # Transpose and reshape to (N, L_q, d_model)
        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3)).reshape(q.shape[0], q.shape[1], -1)


        # Final output projection
        output = self.out_proj(attn_output)


        return output



class AttentionalPooler(nn.Module):
    d_model: int
    context_dim: int
    n_head: int = 8
    n_queries: int = 256
    norm_layer: Callable = nn.LayerNorm

    @nn.compact
    def __call__(self, x):
        N = x.shape[0]

        # Initialize query parameters
        query = self.param('query', nn.initializers.normal(), (self.n_queries, self.d_model))

        # Define the LayerNorm layers
        ln_q = self.norm_layer()
        ln_k = self.norm_layer()

        # Define the multi-head attention layer
        attn = CustomMultiheadAttention(d_model=self.d_model, context_dim=self.context_dim, n_head=self.n_head)

        # Apply layer normalization
        x = ln_k(x)
        q = ln_q(query)

        # Expand query for batch size
        q = jnp.expand_dims(q, 0)
        q = jnp.tile(q, (N, 1, 1))

        # remove some unnecessary logical constraints. For x and q, the constraints added at the input are sufficient, and there is no need to repeat them later.
        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))
        q = nn.with_logical_constraint(q, ("activation_batch", "activation_length", "activation_embed"))

        # calculate attention
        attn_output = attn(q=q, k=x, v=x)

        # add one logical constraint to the output, without adding it to each intermediate step
        attn_output = nn.with_logical_constraint(attn_output, ("activation_batch", "activation_length", "activation_embed"))

        return attn_output


# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, dtype=jnp.float32, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, float):
        grid_size = int(grid_size)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return jnp.asarray(pos_embed, dtype)[None, :, :]


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32, cls_token=False):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  if cls_token:
      pe = jnp.concatenate([jnp.zeros([1, width]), pe], axis=0)
  return jnp.asarray(pe, dtype)[None, :, :]

def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32, cls_token=False):
  if typ == "learn":
    num_token = 1 if cls_token else 0
    return self.param(name,
                      #nn.initializers.variance_scaling(scale=0.3072, distribution="truncated_normal", mode='fan_out'), # timm trunc
                      nn.initializers.normal(stddev=0.02),
                      #nn.initializers.normal(stddev=1/np.sqrt(width)),
                      (1, np.prod(seqshape) + num_token, width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype, cls_token=cls_token)
    #return get_2d_sincos_pos_embed(width, seqshape[0], dtype=dtype, cls_token=cls_token)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0
  dtype: Optional[Dtype] = jnp.float32
  param_dtype: Dtype = jnp.float32

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.with_logical_partitioning(nn.initializers.normal(stddev=0.02), ("embed", "mlp")),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
    )
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

    n, l, d = x.shape  # pylint: disable=unused-variable
    x = nn.Dense(self.mlp_dim or 4 * d, **inits,
                     dtype=self.dtype,
                     param_dtype=self.param_dtype)(x)
    x = x.astype(self.dtype)
    x = nn.gelu(x, approximate=False)
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))


    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = x.astype(self.dtype)
    x = nn.Dense(d,
                 kernel_init=nn.with_logical_partitioning(nn.initializers.variance_scaling(scale=0.3072, distribution="truncated_normal", mode='fan_out'), ("mlp", "embed")), # timm trunc
                 bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
                 dtype=self.dtype,
                 param_dtype=self.param_dtype)(x)
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

    return x


import functools
from typing import (Any, Callable, Optional, Tuple)
from flax.linen.linear import DenseGeneral

from flax.linen.module import compact
from flax.linen.module import merge_param
Array = Any


class LayerScale(nn.Module):
    d:  int = 1024
    init_values: float = 1e-5
    name: str='layer_scale'

    @nn.compact
    def __call__(self, x):

        return x * self.param(
            self.name,
            nn.initializers.constant(self.init_values),
            (self.d),

        )




class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  drop_path: float = 0.0
  init_values: float = None
  use_flash_attn: bool = False
  dtype: Optional[Dtype] = jnp.float32
  param_dtype: Dtype = jnp.float32
  mesh: Any = None
  use_dense_general: bool = False
  scan_mlp: bool = False
  scan_attn: bool = False
  mlp_chunck: int = 128

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    x = x.astype(self.dtype)
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

    y = nn.LayerNorm(
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("norm",)),
        bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), (None,)),
    )(x)
    y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))

    ## hack init func
    y = out["sa"] = common.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        qkv_kernel_init=nn.initializers.normal(stddev=0.02),
        out_kernel_init=nn.initializers.normal(stddev=0.02),
        bias_init=nn.initializers.zeros,
        deterministic=deterministic,
        use_flash_attn=self.use_flash_attn,
        scan_attn=self.scan_attn,
        scan_attn_chunck=self.mlp_chunck,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        mesh=self.mesh,
        use_dense_general=self.use_dense_general
    )(y, y)
    y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))

    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    if self.init_values is not None:
        n, l, d = y.shape
        y = LayerScale(d, init_values=self.init_values, name='ls1')(y)
    y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
    x = out["+sa"] = x + y
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

    y = nn.LayerNorm(dtype=self.dtype,
                     param_dtype=self.param_dtype,
                     scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("norm",)),
                     bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), (None,)),
                     )(x)
    y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))
    mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        dropout=self.dropout,
        param_dtype=self.param_dtype,
        dtype=self.dtype,
    )
    if self.scan_mlp:
        y = out["mlp"] = blockwise_ffn(
            mlp,
            y,
            self.mlp_chunck,
            deterministic,
        )
    else:
        y = out["mlp"] = mlp(y, deterministic)

    y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))

    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    if self.init_values is not None:
        n, l, d = y.shape
        y = LayerScale(d, init_values=self.init_values, name='ls2')(y)
    y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
    y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))

    x = out["+mlp"] = x + y
    x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  drop_path: float = 0.0
  init_values: float = None
  remat_policy: str = "none"
  use_flash_attn: bool = False
  scan_mlp: bool = False
  scan_attn: bool = False
  mlp_chunck: int = 128
  dtype: Optional[Dtype] = jnp.float32
  param_dtype: Dtype = jnp.float32
  mesh: Any = None
  use_dense_general: bool = False

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}
    dpr = [float(x) for x in np.linspace(0, self.drop_path, self.depth)] # drop path decay
    # Input Encoder

    if self.remat_policy != "none":
        if self.remat_policy == "minimal":
            policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
        elif self.remat_policy == "minimal_offloaded":
            policy = jax.checkpoint_policies.offload_dot_with_no_batch_dims(offload_src="device",
                                                                            offload_dst="pinned_host")
        elif self.remat_policy == "minimal_flash":
            policy = jax.checkpoint_policies.save_from_both_policies(
                jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
                jax.checkpoint_policies.save_only_these_names(
                    "context",
                ),
            )
        else:
            assert self.remat_policy == "full", "Remat policy needs to be on list of remat policies"
            policy = None

        #logging.info(f"activation checkpointing {self.remat_policy}")
        BlockLayer = remat(  # pylint: disable=invalid-name
            Encoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(1,)
        )  # "deterministic" is a static argu
    else:
        BlockLayer = Encoder1DBlock

    for lyr in range(self.depth):
      block = BlockLayer(
          name=f"encoderblock_{lyr}",
          mlp_dim=self.mlp_dim, num_heads=self.num_heads, dropout=self.dropout, drop_path=dpr[lyr], use_flash_attn=self.use_flash_attn,
          init_values=self.init_values,
          scan_mlp=self.scan_mlp,
          scan_attn=self.scan_attn,
          mlp_chunck=self.mlp_chunck,
          dtype=self.dtype,
          param_dtype=self.param_dtype,
          mesh=self.mesh,
          use_dense_general=self.use_dense_general
      )
      x, out[f"block{lyr:02d}"] = block(x, deterministic)
    out["pre_ln"] = x  # Alias for last block, but without the number in it.

    return x, out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12

  @nn.compact
  def __call__(self, x):
    # TODO
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param("probe", nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x)

    # TODO: dropout on head?
    y = nn.LayerNorm()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class _Model(nn.Module):
  """ViT model."""

  num_classes: Optional[int] = None
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  rep_size: Union[int, bool] = False
  dropout: float = 0.0
  drop_path: float = 0.0
  pool_type: str = "gap"  # Can also be "map" or "tok"
  head_zeroinit: bool = True
  patch_embeding: str = 'conv'
  init_values: float = None
  remat_policy: str = 'none'
  scan_mlp: bool = False
  scan_attn: bool = False
  mlp_chunck: int = 128
  post_norm: bool = False
  emb_head_bias: bool = True
  mean: Sequence[float] = (0.485, 0.456, 0.406)
  std: Sequence[float] = (0.229, 0.224, 0.225)
  final_drop: float = 0.
  mask_ratio: float = 0.
  mask_mode: str = 'random'
  use_flash_attn: bool = False
  block_size: int = 128
  dtype: Optional[Dtype] = jnp.float32
  param_dtype: Dtype = jnp.float32
  mesh: Any = None
  output_tokens: bool = False
  use_dense_general: bool = False
  ignore_cls: bool = False

  def random_masking(self, x, mask_ratio, rng_mask=None,
                     mask_mode='random', height=14, width=14):

      from jax import vmap
      N, L, D = x.shape  # batch, length, dim
      len_keep = int(L * (1 - mask_ratio))

      if mask_mode == 'random':
          noise = jax.random.uniform(rng_mask, (N, L))
      # for now, only assume mask_ratio 0.75 input. and assume even size input
      # keep naming consistent with previous implementation
      # keep naming consistent with previous implementation
      elif mask_mode == 'square':
          if mask_ratio == 0.5:
              mask_heights = jnp.array([7 // 2, 14], dtype=jnp.int32)
              mask_widths = jnp.array([14, 7 // 2], dtype=jnp.int32)
              rng, rng_mask = jax.random.split(rng_mask, 2)
              index = jax.random.randint(rng, (N,), 0, 2)
              mask_heights = mask_heights[index]
              mask_widths = mask_widths[index]
          elif mask_ratio == 0.25:
              mask_heights = jnp.array([16, 12], dtype=jnp.int32)
              mask_widths = jnp.array([12, 16], dtype=jnp.int32)
              rng, rng_mask = jax.random.split(rng_mask, 2)
              index = jax.random.randint(rng, (N,), 0, 2)
              mask_heights = mask_heights[index]
              mask_widths = mask_widths[index]
          elif mask_ratio == 0.525:
              mask_heights = 11 * jnp.ones((N,), dtype=jnp.int32)
              mask_widths = 11 * jnp.ones((N,), dtype=jnp.int32)
          elif mask_ratio == 0.4375:
              mask_heights = 12 * jnp.ones((N,), dtype=jnp.int32)
              mask_widths = 12 * jnp.ones((N,), dtype=jnp.int32)
          elif mask_ratio == 0.75:
              mask_heights = 7 * jnp.ones((N,), dtype=jnp.int32)
              mask_widths = 7 * jnp.ones((N,), dtype=jnp.int32)
          elif mask_ratio == 0.816:
              mask_heights = jnp.array([4, 6, 9], dtype=jnp.int32)
              mask_widths = jnp.array([9, 6, 4], dtype=jnp.int32)
              rng, rng_mask = jax.random.split(rng_mask, 2)
              index = jax.random.randint(rng, (N,), 0, 3)
              mask_heights = mask_heights[index]
              mask_widths = mask_widths[index]
          elif mask_ratio == 0.875:
              mask_heights = jnp.array([2, 3, 4, 6, 8, 12], dtype=jnp.int32)
              mask_widths = jnp.array([12, 8, 6, 4, 3, 2], dtype=jnp.int32)
              rng, rng_mask = jax.random.split(rng_mask, 2)
              index = jax.random.randint(rng, (N,), 0, 6)
              mask_heights = mask_heights[index]
              mask_widths = mask_widths[index]
          elif mask_ratio == 0.918:
              mask_heights = jnp.array([2, 4, 8], dtype=jnp.int32)
              mask_widths = jnp.array([8, 4, 2], dtype=jnp.int32)
              rng, rng_mask = jax.random.split(rng_mask, 2)
              index = jax.random.randint(rng, (N,), 0, 3)
              mask_heights = mask_heights[index]
              mask_widths = mask_widths[index]

          rng_mask1, rng_mask2 = jax.random.split(rng_mask, 2)

          # cannot use clip here. otherwise half the random value would be zero
          # should randomly choose between plausible height/width
          # start_heights = jax.random.randint(rng_mask1, (N,), 0, height) - mask_heights
          # start_widths = jax.random.randint(rng_mask2, (N,), 0, width) - mask_widths
          # start_heights = jnp.clip(start_heights, 0, height)
          # start_widths = jnp.clip(start_widths, 0, width)

          @functools.partial(vmap, axis_name='batch')
          def generate_start_height(mask_height):
              local_rng = jax.random.fold_in(rng_mask1, jax.lax.axis_index("batch"))
              return jax.random.randint(local_rng, (), 0, height - mask_height)

          @functools.partial(vmap, axis_name='batch')
          def generate_start_width(mask_width):
              local_rng = jax.random.fold_in(rng_mask2, jax.lax.axis_index("batch"))
              return jax.random.randint(local_rng, (), 0, width - mask_width)

          start_heights = generate_start_height(mask_heights)
          start_widths = generate_start_width(mask_widths)

          def _window_mask(destination_box: jax.Array,
                           size: Tuple[int, int]) -> jnp.ndarray:
              """Mask a part of the image."""
              # copied from mixup.py. slightly modified.
              height_offset, width_offset, h, w = destination_box
              h_range = jnp.reshape(jnp.arange(size[0]), [size[0], 1])
              w_range = jnp.reshape(jnp.arange(size[1]), [1, size[1]])
              return jnp.logical_and(
                  jnp.logical_and(height_offset <= h_range,
                                  h_range < height_offset + h),
                  jnp.logical_and(width_offset <= w_range,
                                  w_range < width_offset + w)).astype(jnp.int32)

          @functools.partial(vmap, axis_name='batch')
          def square_mask(start_height, start_width, mask_height, mask_width):
              return _window_mask((start_height, start_width, mask_height, mask_width), (height, width))

          noise = square_mask(start_heights, start_widths, mask_heights, mask_widths)
          noise = jnp.logical_not(noise).astype(int)
          noise = noise.reshape((N, L))
      elif mask_mode == 'per2x2_random_grid':
          noises = jnp.ones((N, L), dtype=int)
          # TODO: add support for mask_ratio = 0.5, and 0.875
          noises = noises.reshape(N, height, width)

          if mask_ratio == 0.5:
              noises = noises.reshape(N, height // 2, 2, width // 2, 2)
              noises = jnp.transpose(noises, (0, 2, 4, 1, 3))
              noises = noises.reshape(N, 4, height * width // 4)

              choice = jnp.array([0, 1, 2, 3], dtype=jnp.int32)
              rng, rng_mask = jax.random.split(rng_mask, 2)

              @functools.partial(vmap, axis_name='batch')
              def per2x2_random_grid_indices(mask):
                  local_rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
                  return jax.random.choice(local_rng, choice, shape=(2,), replace=False)

              indices = per2x2_random_grid_indices(jnp.ones((N * height * width // 4,)))
              indices = indices.reshape(N, height * width // 4, 2)

              @vmap
              def per2x2_random_grid_mask(mask, index):
                  # note that this at+set returns a new copy! it is not a in-place operation!
                  mask = mask.at[index[:, 0], jnp.arange(height * width // 4)].set(0)
                  mask = mask.at[index[:, 1], jnp.arange(height * width // 4)].set(0)
                  return mask

              noise = per2x2_random_grid_mask(noises, indices)
              noise = noise.reshape((N, 2, 2, height // 2, width // 2))
              noise = jnp.transpose(noise, (0, 3, 1, 4, 2))
              noise = noise.reshape(N, height, width)
              noise = noise.reshape((N, L))
          elif mask_ratio == 0.75 or mask_ratio == 0.25:
              noises = noises.reshape(N, height // 2, 2, width // 2, 2)
              noises = jnp.transpose(noises, (0, 2, 4, 1, 3))
              noises = noises.reshape(N, 4, height * width // 4)

              indices = jax.random.randint(rng_mask, (N, height * width // 4,), 0, 4)

              @vmap
              def per2x2_random_grid_mask(mask, index):
                  # note that this at+set returns a new copy! it is not a in-place operation!
                  return mask.at[index, jnp.arange(height * width // 4)].set(0)
                  # return mask

              noise = per2x2_random_grid_mask(noises, indices)
              if mask_ratio == 0.25:
                  noise = jnp.logical_not(noise).astype(int)
              noise = noise.reshape((N, 2, 2, height // 2, width // 2))
              noise = jnp.transpose(noise, (0, 3, 1, 4, 2))
              noise = noise.reshape(N, height, width)
              noise = noise.reshape((N, L))
      #     noise = jax.random.uniform(rng_mask, (N, L))

      # sort noise for each sample
      ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
      ids_restore = jnp.argsort(ids_shuffle, axis=1)

      # keep the first subset
      ids_keep = ids_shuffle[:, :len_keep]
      # x_masked = batched_gather(x, ids_keep)

      x_masked = jnp.take_along_axis(x, ids_keep[:, :, None], 1)

      # generate the binary mask: 0 is keep, 1 is remove
      mask = jnp.ones((N, L))
      mask = mask.at[:, :len_keep].set(0)

      # mask = batched_gather(mask, ids_restore)
      mask = jnp.take_along_axis(mask, ids_restore, 1)

      return x_masked, mask, ids_restore

  def _global_pool(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if self.pool_type == 'avg':
        # Use JAX's mean function for better parallelism
        if self.ignore_cls:
            pooled = jnp.mean(x[:, 1:], axis=1)
            tokens = x[:, 1:]
        else:
            pooled = jnp.mean(x, axis=1)
            tokens = x
    elif self.pool_type == 'tok':
        pooled = x[:, 0]
        tokens = x[:, 1:]
    else:
        pooled = tokens = x
    
    return pooled, tokens

  @nn.compact
  def __call__(self, image, *, train=False):
    out = {}
    if self.post_norm:
        mean = jnp.asarray(
           self.mean)[None, None, None, :]
        std = jnp.asarray(
            self.std)[None, None, None, :]
        image = (image - mean) / std
    if self.patch_embeding == 'conv':
        # Patch extraction
        x = out["stem"] = nn.Conv(
            self.width, self.patch_size, strides=self.patch_size,
            kernel_init=nn.with_logical_partitioning(nn.initializers.kaiming_uniform(), (None, None, None, None)),
            use_bias=self.emb_head_bias,
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            padding="VALID", name="embedding")(image)

        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
    elif self.patch_embeding == 'stem':
        x = image
        width_list = [96, 192, 384]
        stride_list = [1, 2, 2]
        kernel_list = [3, 3, 3]
        for i in range(3):
            x = out["stem"] = nn.Conv(
                width_list[i], (kernel_list[i], kernel_list[i]),
                strides=stride_list[i],
                kernel_init=nn.initializers.kaiming_uniform(),
                bias_init=nn.initializers.zeros,
                padding="SAME", name=f"stem_{i}")(x)
            x = nn.LayerNorm(name=f"stem_norm_{i}")(x)
            x = nn.gelu(x)

        x = out["stem"] = nn.Conv(
                self.width, (1, 1), strides=1,
                kernel_init=nn.initializers.kaiming_uniform(),
                bias_init=nn.initializers.zeros,
                padding="SAME", name="embedding")(x)
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])
    else:
        p = self.patch_size[0]
        h = w = image.shape[2] // p
        x = image.reshape((image.shape[0], h, p, w, p, 3))
        x = jnp.einsum('nhpwqc->nhwpqc', x)
        x = x.reshape((image.shape[0], h * w, p ** 2 * 3))
        x = out["stem"] = nn.Dense(self.width, name="embedding")(x)
        n, l,  c = x.shape



    cls = self.param("cls", nn.with_logical_partitioning(nn.initializers.normal(1e-6), (None,)), (1, 1, c), x.dtype)
    x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    # Add posemb before adding extra token.
    x = out["with_posemb"] = x + get_posemb(
        self, self.posemb, (h, w), c, "pos_embedding", x.dtype, cls_token=True)


    x = x.astype(self.dtype)
    x = nn.with_logical_constraint(x, (
        "activation_batch", "activation_length", "activation_embed"))
    x = nn.Dropout(rate=self.dropout)(x, not train)
    if self.mask_ratio > 0 and train:
        cls_token = x[:, :1]
        rng_mask = self.make_rng('random_mask')
        x, _, _ = self.random_masking(
            x[:, 1:],
            mask_ratio=self.mask_ratio,
            mask_mode=self.mask_mode,
            rng_mask=rng_mask)
        x = jnp.concatenate([cls_token, x], axis=1)
    x = nn.with_logical_constraint(x, (
        "activation_batch", "activation_length", "activation_embed"))
    if self.ignore_cls:
        x  = x[:, 1:]
    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        scan_mlp=self.scan_mlp,
        scan_attn=self.scan_attn,
        mlp_chunck=self.mlp_chunck,
        dropout=self.dropout,
        drop_path=self.drop_path,
        init_values = self.init_values,
        remat_policy=self.remat_policy,
        use_flash_attn=self.use_flash_attn,
        dtype=self.dtype,
        param_dtype=self.param_dtype,
        mesh=self.mesh,
        use_dense_general=self.use_dense_general,
        name="Transformer")(
            x, deterministic=not train)
    encoded = out["encoded"] = x
    x = nn.with_logical_constraint(x, (
        "activation_batch", "activation_length", "activation_embed"))
    # x = jnp.mean(x[:, 1:], axis=1)
    # x = x[:, 0]
    if self.pool_type == "map":
      x = out["head_input"] = MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
    elif self.pool_type == "gap":
      if self.ignore_cls:
          x = jnp.mean(x, axis=1)
      else:
         x  = jnp.mean(x[:, 1:], axis=1)
      x = out["head_input"] = nn.LayerNorm(name="encoder_norm",
                                           dtype=self.dtype,
                                           scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(),
                                                                                   ("norm",)),
                                           bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(),
                                                                                  (None,)),
                                           param_dtype=self.param_dtype,)(x)
    #   encoded = encoded[:, 1:]
    elif self.pool_type == "0":
      x = out["head_input"] = x[:, 0]
    elif self.pool_type == "tok":
      x =  nn.LayerNorm(name="encoder_norm",dtype=self.dtype,
                        scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("norm",)),
                        bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), (None,)),
                        param_dtype=self.param_dtype)(x)
      x = out["head_input"] = x[:, 0]
    #   encoded = encoded[:, 1:]
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")
    
    tokens = encoded[:, 1:]

    out["pre_logits"] = x

    if self.num_classes:

      head = nn.Dense(self.num_classes, name="head",
                      kernel_init=nn.with_logical_partitioning(nn.initializers.normal(stddev=0.02), ("embed", "vocab")),
                      bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
                      use_bias=self.emb_head_bias,
                      dtype=jnp.float32,
                      param_dtype=self.param_dtype,
                     )
      x = nn.with_logical_constraint(x, (
          "activation_batch", "activation_embed"))
      x = nn.Dropout(rate=self.final_drop)(x, not train)

      x = out["logits"] = head(x)
      x = nn.with_logical_constraint(x, ("activation_batch", "activation_embed"))
    if self.output_tokens:
        return x, tokens
    
    return x


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _Model(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
  """Converts a string like "B" or "B/32" into a params dict."""
  if variant is None:
    return {}

  v, patch = variant, {}
  if "/" in variant:
    v, patch = variant.split("/")
    patch = {"patch_size": (int(patch), int(patch))}

  return {
      # pylint:disable=line-too-long
      # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
      "width": {"mu": 32, "Ti": 192, "S": 384, "M": 512, "B": 768, "L": 1024, "So400m": 1152, "H": 1280, "g": 1408, "g-opt": 1536, "G": 1664, "G-opt": 1536, "e": 1792}[v],
      "depth": {"mu": 1, "Ti": 12, "S": 12, "M": 12, "B": 12, "L": 24, "So400m": 27, "H": 32, "g": 40, "g-opt": 40, "G": 48, "G-opt": 48, "e": 56}[v],
      "mlp_dim": {"mu": 128, "Ti": 768, "S": 1536, "M": 2048, "B": 3072, "L": 4096, "So400m": 4304, "H": 5120, "g": 6144, "g-opt": 6144, "G": 8192, "G-opt": 8192, "e": 15360}[v],
      "num_heads": {"mu": 2, "Ti": 3, "S": 6, "M": 8, "B": 12, "L": 16, "So400m": 16, "H": 16, "g": 16, "g-opt": 16, "G": 16, "G-opt": 16, "e": 16}[v],
      # pylint:enable=line-too-long
      **patch
  }


def resample_posemb(old, new):
  """This function implements "high-res finetuning" for transformer models."""
  # Rescale the grid of position embeddings. Param shape is (1,N,1024)
  if old.shape == new.shape:
    return old

  # extract cls
  cls_pos = old[:, 0:1, :]
  old = old[:, 1: , :]
  new = new[:, 1:, :]

  logging.info("ViT: resize %s to %s", old.shape, new.shape)
  gs_old = int(np.sqrt(old.shape[1]))
  gs_new = int(np.sqrt(new.shape[1]))
  logging.info("ViT: grid-size from %s to %s", gs_old, gs_new)
  grid = old.reshape(gs_old, gs_old, -1)

  zoom = (gs_new/gs_old, gs_new/gs_old, 1)
  grid = scipy.ndimage.zoom(grid, zoom, order=1)
  grid = grid.reshape(1, gs_new*gs_new, -1)

  #add cls
  grid = jnp.concatenate([cls_pos, grid], axis=1)
  return jnp.array(grid)


def fix_old_checkpoints(params):
  """Fix small bwd incompat that can't be resolved with names in model def."""

  params = flax.core.unfreeze(
      flax.training.checkpoints.convert_pre_linen(params))

  # Original ViT paper variant had posemb in a module:
  if "posembed_input" in params["Transformer"]:
    logging.info("ViT: Loading and fixing VERY old posemb")
    posemb = params["Transformer"].pop("posembed_input")
    params["pos_embedding"] = posemb["pos_embedding"]

  # Widely used version before 2022 had posemb in Encoder:
  if "pos_embedding" in params["Transformer"]:
    logging.info("ViT: Loading and fixing old posemb")
    params["pos_embedding"] = params["Transformer"].pop("pos_embedding")

  # Old vit.py used to first concat [cls] token, then add posemb.
  # This means a B/32@224px would have 7x7+1 posembs. This is useless and clumsy
  # so we changed to add posemb then concat [cls]. We can recover the old
  # checkpoint by manually summing [cls] token and its posemb entry.
  if "pos_embedding" in params:
    pe = params["pos_embedding"]
    if int(np.sqrt(pe.shape[1])) ** 2 + 1 == int(pe.shape[1]):
      logging.info("ViT: Loading and fixing combined cls+posemb")
      pe_cls, params["pos_embedding"] = pe[:, :1], pe[:, 1:]
      if "cls" in params:
        params["cls"] += pe_cls

  # MAP-head variants during ViT-G development had it inlined:
  if "probe" in params:
    params["MAPHead_0"] = {
        k: params.pop(k) for k in
        ["probe", "MlpBlock_0", "MultiHeadDotProductAttention_0", "LayerNorm_0"]
    }

  return params


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  del model_cfg

  init_file = VANITY_NAMES.get(init_file, init_file)
  restored_params = utils.load_params(None, init_file)

  #restored_params = fix_old_checkpoints(restored_params)

  # possibly use the random init for some of the params (such as, the head).
  restored_params = common.merge_params(restored_params, init_params, dont_load=dont_load)


  # resample posemb if needed.
  if init_params and "pos_embedding" in init_params:
    restored_params["pos_embedding"] = resample_posemb(
        old=restored_params["pos_embedding"],
        new=init_params["pos_embedding"])

  if 'pos_embedding'  in dont_load:
      logging.info('fixed pos_embedding cannot be stored, re-intialized needed')
      _, l, c = init_params["pos_embedding"].shape
      h, w = (l-1)**.5, (l-1)**.5
      #restored_params['pos_embedding'] = get_2d_sincos_pos_embed(c, h, cls_token=True)
      restored_params['pos_embedding'] = posemb_sincos_2d( h, w, c, cls_token=True)

  from helpers.utils import recover_dtype
  restored_params =  jax.tree.map(recover_dtype, restored_params)
  return restored_params


# Shortcut names for some canonical paper checkpoints:
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # pylint: disable=line-too-long
    # Recommended models from https://arxiv.org/abs/2106.10270
    # Many more models at https://github.com/google-research/vision_transformer
    "howto-i21k-Ti/16": "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-S/32": "gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-S/16": "gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-B/32": "gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/16": "gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/8": "gs://vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-L/16": "gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",

    # Better plain vit-s16 baselines from https://arxiv.org/abs/2205.01580
    "i1k-s16-90ep": "gs://big_vision/vit_s16_i1k_90ep.npz",
    "i1k-s16-150ep": "gs://big_vision/vit_s16_i1k_150ep.npz",
    "i1k-s16-300ep": "gs://big_vision/vit_s16_i1k_300ep.npz",

    # DeiT-3 checkpoints from https://github.com/facebookresearch/deit/blob/main/README_revenge.md
    # First layer converted to take inputs in [-1,1]
    "deit3_S_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_1k.npz",
    "deit3_S_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_21k.npz",
    "deit3_S_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_1k.npz",
    "deit3_S_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_21k.npz",
    "deit3_B_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_1k.npz",
    "deit3_B_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_21k.npz",
    "deit3_B_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_1k.npz",
    "deit3_B_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_21k.npz",
    "deit3_L_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_1k.npz",
    "deit3_L_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_21k.npz",
    "deit3_L_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_1k.npz",
    "deit3_L_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_21k.npz",
    # pylint: disable=line-too-long
    # pylint: enable=line-too-long
}
