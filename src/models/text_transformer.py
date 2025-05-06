# #Copyright @2023 Xianhang Li
#
# # This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# # Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.

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


import functools
import sys
from typing import Any, Callable, Optional, Tuple


import flax
import flax.training.checkpoints
from flax import linen as nn
from flax.linen import DenseGeneral
from flax.linen.module import compact
from flax.linen.module import merge_param
from flax.linen.partitioning import remat
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash_attention
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.shard_map import shard_map
import numpy as np

# internal imports from src.models.bpt import blockwise_ffn, blockwise_attn
from src.models.common import DropPath

Array = Any
Dtype = Any  # this could be a real type?

def posemb_sincos_1d(
        max_len,
        width,
        min_scale=1.,
        max_scale=10_000.,
        dtype=jnp.float32,
        cls_token=False):
    """Follows the MoCo v3 logic."""
    d_feature = width
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe, dtype=dtype)


def get_posemb(
        self,
        typ,
        max_len,
        width,
        name,
        dtype=jnp.float32,
        cls_token=False):
    if typ == "learn":
        num_token = 1 if cls_token else 0
        return self.param(name,
                          # nn.initializers.variance_scaling(scale=0.3072,
                          # distribution="truncated_normal", mode='fan_out'), #
                          # timm trunc
                          nn.with_logical_partitioning(nn.initializers.normal(stddev=0.01), (None, None, 'embed')),
                          (1, max_len, width), dtype)
    elif typ == "sincos1d":
        return posemb_sincos_1d(
            max_len,
            width,
            dtype=dtype,
            cls_token=cls_token)
      
    else:
        raise ValueError(f"Unknown posemb type: {typ}")


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    dropout: float = 0.0
    fc_init: Callable = nn.initializers.xavier_uniform()
    proj_init: Callable = nn.initializers.xavier_uniform()
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic=True):
        """Applies Transformer MlpBlock module."""


        n, l, d = x.shape  # pylint: disable=unused-variable
        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

        x = nn.Dense(self.mlp_dim or 4 * d,
                     kernel_init=nn.with_logical_partitioning(self.fc_init, ("embed", "mlp")),
                     bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
                     dtype=self.dtype,
                     param_dtype=self.param_dtype,
                     )(x)
        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))
        x = x.astype(self.dtype)

        x = nn.gelu(x, approximate=True)
        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = x.astype(self.dtype)
        x = nn.Dense(d,
                     kernel_init=nn.with_logical_partitioning(self.proj_init, ("mlp", "embed")),
                     bias_init=nn.with_logical_partitioning(nn.initializers.zeros, (None,)),
                     dtype=self.dtype,
                     param_dtype=self.param_dtype,
                     )(x)
        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

        return x


class MultiHeadDotProductAttention(nn.MultiHeadDotProductAttention):

    attn_kernel_init: Callable = nn.initializers.normal(stddev=0.01)
    proj_kernel_init: Callable = nn.initializers.normal(stddev=0.01)
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Any  = None
    use_dense_general: bool = False
    scan_attn: bool = False
    scan_attn_chunck: int = 128

    @compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Optional[Array] = None,
                 deterministic: Optional[bool] = None):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        Args:
          inputs_q: input queries of shape
            `[batch_sizes..., length, features]`.
          inputs_kv: key/values of shape
            `[batch_sizes..., length, features]`.
          mask: attention mask of shape
            `[batch_sizes..., num_heads, query_length, key/value_length]`.
            Attention weights are masked out if their corresponding mask value
            is `False`.
          deterministic: if false, the attention weight is masked randomly
            using dropout, whereas if true, the attention weights
            are deterministic.

        Returns:
          output of shape `[batch_sizes..., length, features]`.
        """
        inputs_q = nn.with_logical_constraint(inputs_q, ("activation_batch", "activation_length", "activation_embed"))
        inputs_kv = nn.with_logical_constraint(inputs_kv, ("activation_batch", "activation_length", "activation_embed"))

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, (
            'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        if self.use_dense_general:
            dense = functools.partial(DenseGeneral,
                                      axis=-1,
                                      dtype=self.dtype,
                                      param_dtype=self.param_dtype,
                                      features=(self.num_heads, head_dim),
                                      kernel_init=nn.with_logical_partitioning(self.attn_kernel_init,
                                                                               ("embed", "heads")),
                                      bias_init=nn.with_logical_partitioning(self.bias_init, (None,)),
                                      use_bias=self.use_bias,
                                      precision=self.precision)
            # project inputs_q to multi-headed q/k/v
            # dimensions are then [batch..., length, n_heads, n_features_per_head]
            query, key, value = (dense(name='query')(inputs_q),
                                 dense(name='key')(inputs_kv),
                                 dense(name='value')(inputs_kv))
        else:
            q_dense = nn.Dense(qkv_features,
                     name="query",
                     dtype=self.dtype,
                     param_dtype=self.param_dtype,
                     use_bias=self.use_bias,
                     kernel_init=nn.with_logical_partitioning(self.attn_kernel_init, ("embed", "mlp")),
                     bias_init=nn.with_logical_partitioning(self.bias_init, (None,))
                     )
            k_dense = nn.Dense(qkv_features,
                               name="key",
                               dtype=self.dtype,
                               param_dtype=self.param_dtype,
                               use_bias=self.use_bias,
                               kernel_init=nn.with_logical_partitioning(self.attn_kernel_init, ("embed", "mlp")),
                               bias_init=nn.with_logical_partitioning(self.bias_init, (None,))
                               )
            v_dense = nn.Dense(qkv_features,
                               name="value",
                               dtype=self.dtype,
                               param_dtype=self.param_dtype,
                               use_bias=self.use_bias,
                               kernel_init=nn.with_logical_partitioning(self.attn_kernel_init, ("embed", "mlp")),
                               bias_init=nn.with_logical_partitioning(self.bias_init, (None,))
                               )

            query, key, value = (q_dense(inputs_q),
                                 k_dense(inputs_kv),
                                 v_dense(inputs_kv))
            query = nn.with_logical_constraint(query, ("activation_batch", "activation_length", "activation_embed"))
            key = nn.with_logical_constraint(key, ("activation_batch", "activation_length", "activation_embed"))
            value = nn.with_logical_constraint(value, ("activation_batch", "activation_length", "activation_embed"))

            query = self._split_heads(query, num_heads=self.num_heads, head_dim=head_dim)
            key = self._split_heads(key,num_heads=self.num_heads, head_dim=head_dim)
            value = self._split_heads(value,num_heads=self.num_heads, head_dim=head_dim)

        query = nn.with_logical_constraint(query, (
        "activation_batch", "activation_length", "activation_heads", "activation_kv"))
        key = nn.with_logical_constraint(key,
                                         ("activation_batch", "activation_length", "activation_heads", "activation_kv"))
        value = nn.with_logical_constraint(value, (
        "activation_batch", "activation_length", "activation_heads", "activation_kv"))

        dropout_rng = None
        if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
            m_deterministic = merge_param('deterministic', self.deterministic,
                                          deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng('dropout')
        else:
            m_deterministic = True
        # apply attention
        query = query.astype(self.dtype)
        key = key.astype(self.dtype)
        value = value.astype(self.dtype)

        if (not self.use_flash_attn) and (not self.scan_attn):
            x = self.attention_fn(
                query,
                key,
                value,
                mask=mask,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=m_deterministic,
                dtype=self.dtype,
                precision=self.precision)  # pytype: disable=wrong-keyword-args
        elif self.scan_attn:
            x = blockwise_attn(
                query,
                key,
                value,
                causal=False,
                dropout_rng=dropout_rng,
                query_chunk_size=self.scan_attn_chunck,
                key_chunk_size=self.scan_attn_chunck,
                deterministic=m_deterministic,
                dtype=self.dtype,
                precision=self.precision)

        else:
            x = self._tpu_flash_attention(
                query, key, value, decoder_segment_ids=None, head_dim=head_dim
            )
        x = nn.with_logical_constraint(x,
                                       ("activation_batch", "activation_length", "activation_heads", "activation_kv"))
        x = x.astype(self.dtype)
        if not self.use_dense_general:
            x = self._merge_heads(x, embed_dim=qkv_features)
            x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))
            x = nn.Dense(features,
                         name="out",
                         dtype=self.dtype,
                         param_dtype=self.param_dtype,
                         use_bias=self.use_bias,
                         kernel_init=nn.with_logical_partitioning(self.proj_kernel_init,
                                                                  ("mlp", "embed")),
                         bias_init=nn.with_logical_partitioning(self.bias_init, (None,)))(x)
        else:
        # back to the original inputs dimensions
            x = DenseGeneral(features=features,
                             axis=(-2, -1),
                             kernel_init=nn.with_logical_partitioning(self.proj_kernel_init, ("heads", "embed"), ),
                             bias_init=nn.with_logical_partitioning(self.bias_init, (None,)),
                             use_bias=self.use_bias,
                             dtype=self.dtype,
                             param_dtype=self.param_dtype,
                             precision=self.precision,
                             name='out')(x)

        return x

    def _split_heads(self, hidden_states, num_heads, head_dim):
        return hidden_states.reshape(hidden_states.shape[:2] + (num_heads, head_dim))

    def _merge_heads(self, hidden_states, embed_dim):
        return hidden_states.reshape(hidden_states.shape[:2] + (embed_dim,))

    def _tpu_flash_attention(self, query: Array, key: Array, value: Array, decoder_segment_ids: Array | None, head_dim: int) -> Array:
        """TPU Flash Attention."""
        assert  self.mesh, 'need specify device mesh to use flash attention'
        # Transpose to ('batch', 'heads', 'length', 'kv')
        query = jnp.transpose(query, axes=(0, 2, 1, 3))
        key = jnp.transpose(key, axes=(0, 2, 1, 3))
        value = jnp.transpose(value, axes=(0, 2, 1, 3))

        if decoder_segment_ids is not None:
            decoder_segment_ids = splash_attention_kernel.SegmentIds(decoder_segment_ids, decoder_segment_ids)
        axis_names = jax.sharding.PartitionSpec(("data", "fsdp"),
                                                 None,
                                                 None,
                                                 None)
       

        segment_axis_names = jax.sharding.PartitionSpec(
            (("data", "fsdp"), "tensor")
        )
        @functools.partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(
                    axis_names,
                    axis_names,
                    axis_names,
                    segment_axis_names,
            ),
            out_specs=axis_names,
            check_rep=False,
        )
        def wrap_flash_attention(query, key, value, decoder_segment_ids, head_dim):
            if decoder_segment_ids is not None:
                assert (
                        query.shape[2]
                        == decoder_segment_ids.q.shape[1]
                ), 'Sharding along sequence dimension not allowed in flash attention'

            return tpu_flash_attention.flash_attention(
                query,
                key,
                value,
                causal=self.casual_mask,
                segment_ids=decoder_segment_ids,
                sm_scale=head_dim ** -0.5,
                block_sizes=tpu_flash_attention.BlockSizes(
                    block_q=min(128, query.shape[2]),
                    block_k_major=min(128, key.shape[2]),
                    block_k=min(128, key.shape[2]),
                    block_b=min(1, query.shape[0]),
                    block_q_major_dkv=min(128, query.shape[2]),
                    block_k_major_dkv=min(128, key.shape[2]),
                    block_q_dkv=min(128, query.shape[2]),
                    block_k_dkv=min(128, key.shape[2]),
                    block_q_dq=min(128, query.shape[2]),
                    block_k_dq=min(128, key.shape[2]),
                    block_k_major_dq=min(128, key.shape[2]),
                ))
        devices_in_data_fsdp = self.mesh.shape["data"] * self.mesh.shape["fsdp"]
        assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
            "Batch dimension should be shardable among the devices in data and fsdp" " axis"
        )
        x = wrap_flash_attention(query, key, value, decoder_segment_ids, head_dim)
        x = jnp.transpose(x, axes=(0, 2, 1, 3))
        return x


class Encoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    drop_path: float = 0.0
    depth: int = 12
    casual_mask: bool = False
    use_flash_attn: bool = False
    scan_mlp: bool = False
    scan_attn: bool = False
    mlp_chunck: int = 128
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Any = None
    fusion_style: str = "cross_attn"
    li: int = 0
    lt: int = 0
    use_dense_general: bool = False

    @nn.compact
    def __call__(self, x, deterministic=True):
        width = x.shape[-1]
        init_std = {
            'proj': (width ** -0.5) * ((2 * self.depth) ** -0.5),
            'attn': width ** -0.5,
            'fc': (2 * width) ** -0.5
        }
        out = {}
        if self.casual_mask:
            if self.fusion_style=="cross_attn":
                mask = flax.linen.attention.make_causal_mask(x[:, :, 0]) # force input shape as bz, length
                mask = jnp.broadcast_to(mask, shape=(x.shape[0], self.num_heads, x.shape[1], x.shape[1]))
            else:
                li = self.li
                lt = self.lt
                l = li + lt 
                # Generate causal mask for the text part (lt)
                causal_mask = nn.attention.make_causal_mask(x[:, li:, 0])  # Shape: (batch_size, 1, lt, lt)

                # Step 4: Create the prefix mask (li part), where all elements in image (prefix) can attend to each other
                prefix_mask = jnp.ones((x.shape[0], li, li), dtype=bool)  # Shape: (batch_size, li, li)

                # Step 5: Combine prefix and target parts into a full mask
                # Create a full mask with the shape (batch_size, l, l)
                mask = jnp.zeros((x.shape[0], l, l), dtype=bool)

                # Place the prefix mask in the top-left corner of the full mask (image attends to image)
                mask = mask.at[:, :li, :li].set(prefix_mask)

                # Place the causal mask in the bottom-right corner of the full mask (text attends causally to itself)
                mask = mask.at[:, li:, li:].set(jnp.squeeze(causal_mask, axis=1))

                # Ensure the text can attend to the image embeddings (bottom-left corner)
                mask = mask.at[:, li:, :li].set(True)  # Text attends to all image embeddings
                # Step 6: Broadcast the full mask to match multi-head attention dimensions
                # Initially the num_heads dimension is 1, and then we broadcast it to the actual num_heads size
                mask = jnp.expand_dims(mask, axis=1)  # Shape: (batch_size, 1, l, l)
                mask = jnp.broadcast_to(mask, shape=(x.shape[0], self.num_heads, l, l))


        x = x.astype(self.dtype)
        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

        y = nn.LayerNorm(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("norm",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), (None,)),
           )(x)
        y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))

        y = out["sa"] = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            attn_kernel_init=nn.initializers.normal(stddev=init_std['attn']),
            proj_kernel_init=nn.initializers.normal(stddev=init_std['proj']),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            deterministic=deterministic,
            use_flash_attn=self.use_flash_attn,
            scan_attn=self.scan_attn,
            scan_attn_chunck=self.mlp_chunck,
            mesh=self.mesh,
            use_dense_general=self.use_dense_general
        )(y, y, mask=mask if self.casual_mask else None)
        y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
        x = out["+sa"] = x + y
        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

        y = nn.LayerNorm(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("norm",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), (None,)),
        )(x)
        y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))
        mlp = MlpBlock(
            mlp_dim=self.mlp_dim, dropout=self.dropout,
            fc_init=nn.initializers.normal(stddev=init_std['fc']),
            proj_init=nn.initializers.normal(stddev=init_std['proj']),
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
    remat_policy: str = "none"
    casual_mask: bool = False
    scan_mlp: bool = False
    scan_attn: bool = False
    mlp_chunck: int = 128
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Any = None
    fusion_style: str = "cross_attn"
    li: int = 0
    lt: int = 0
    use_dense_general: bool = False

    @nn.compact
    def __call__(self, x, deterministic=True):
        out = {}
        dpr = [
            float(x) for x in np.linspace(
                0,
                self.drop_path,
                self.depth)]  # drop path decay
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

            BlockLayer = remat(  # pylint: disable=invalid-name
                Encoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(1,)
            )  # "deterministic" is a static argu
        else:
            BlockLayer = Encoder1DBlock

        for lyr in range(self.depth):
            x, out[f"block{lyr:02d}"] = BlockLayer(
                name=f"encoderblock_{lyr}",
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                drop_path=dpr[lyr],
                casual_mask=self.casual_mask,
                use_flash_attn=self.use_flash_attn,
                scan_mlp=self.scan_mlp,
                scan_attn=self.scan_attn,
                mlp_chunck=self.mlp_chunck,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                fusion_style=self.fusion_style,
                li=self.li,
                lt=self.lt,
                use_dense_general=self.use_dense_general,
                mesh=self.mesh
            )(x, deterministic)
         # x, out[f"block{lyr:02d}"] = block(x, deterministic)
        # Alias for last block, but without the number in it.
        out["pre_ln"] = x

        return x, out


class _Model(nn.Module):
    """ViT model."""

    num_classes: Optional[int] = None
    width: int = 512
    depth: int = 12
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    posemb: str = "learn"  # Can also be "sincos2d"
    pool_type: str = "last"  # Can also be "map" or "tok"
    vocab_size: int = 32000
    head_zeroinit: bool = False
    drop_path: float = 0.0
    remat_policy: str = 'none'
    casual_mask: bool = False
    use_flash_attn: bool = False
    scan_mlp: bool = False
    scan_attn: bool = False
    mlp_chunck: int = 128
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Any = None
    embed_cls: bool = False
    output_tokens: bool = False
    use_dense_general: bool = False

    def text_global_pool(self, x, text: Optional[jnp.ndarray] = None, pool_type: str = 'argmax'):
        if pool_type == 'first':
            pooled, tokens = x[:, 0], x[:, 1:]
        elif pool_type == 'last':
            pooled, tokens = x[:, -1], x[:, :-1]
        elif pool_type == 'argmax':
            assert text is not None
            pooled, tokens = x[jnp.arange(x.shape[0]), jnp.argmax(text, axis=-1)], x
        else:
            pooled = tokens = x

        return pooled, tokens

    @nn.compact
    def __call__(self, text, *, train=False, mask_ratio=0):
        out = {}

        embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.width,
            dtype=jnp.float32, # for logit training stability
            param_dtype=self.param_dtype,
            embedding_init=nn.with_logical_partitioning(nn.initializers.normal(
                stddev=0.02), ('vocab', 'embed')))
        x = out['embedded'] = embedding(text.astype("int32"))

        n, l, d = x.shape  # pylint: disable=unused-variable

        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))
        # Add posemb before adding extra token.
        x = x.astype(self.param_dtype) #compatiable with positional embedding

        x = x + get_posemb(
            self, self.posemb, l, d, "pos_embedding", self.param_dtype, cls_token=True)
        x = x.astype(self.dtype)

        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))
        

        x = nn.Dropout(rate=self.dropout)(x, not train)
        encoder_blocks = Encoder(
            depth=self.depth,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            drop_path=self.drop_path,
            remat_policy=self.remat_policy,
            casual_mask=self.casual_mask,
            scan_mlp=self.scan_mlp,
            scan_attn=self.scan_attn,
            mlp_chunck=self.mlp_chunck,
            use_flash_attn=self.use_flash_attn,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            mesh=self.mesh,
            use_dense_general=self.use_dense_general,
            name="Transformer")

        x, out["encoder"] = encoder_blocks(
            x, deterministic=not train)
        out["encoded"] = x
        # ========== BEGIN MODIFICATION ==========
        """
        ********* MODIFICATION NOTICE *********
        This section of the code has been modified
        """
        tokens=out["encoded"][:, :-1]
        x = out["norm"] = nn.LayerNorm(name="encoder_norm")(x)
        x = out["head_input"] = x[:, -1, :]
        # ========== END MODIFICATION ==========

        if self.num_classes:
            x = nn.with_logical_constraint(x, (
            "activation_batch", "activation_embed"))

            head = nn.Dense(
                self.num_classes,
                name="head",
                use_bias=False,
                dtype=jnp.float32, # for logit training stability
                param_dtype=self.param_dtype,
                kernel_init=nn.with_logical_partitioning(
                    nn.initializers.normal(stddev=self.width ** -0.5), ("embed", "vocab")))

            x = out["logits"] = head(x)
            x = nn.with_logical_constraint(x, (
            "activation_batch", "activation_embed"))

        if self.output_tokens:
            return x, tokens

        return x

    def random_masking(self, x, mask_ratio, rng_mask=None):

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = jax.random.uniform(rng_mask, (N, L))

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        #x_masked = batched_gather(x, ids_keep)

        x_masked = jnp.take_along_axis(x, ids_keep[:, :, None], 1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = jnp.ones((N, L))
        mask = mask.at[:, :len_keep].set(0)

        #mask = batched_gather(mask, ids_restore)
        mask = jnp.take_along_axis(mask, ids_restore, 1)
        return x_masked, mask, ids_restore


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
        "width": {"Ti": 192, "S": 384, "M": 512, "B": 512, "L": 768, "So400m": 1152, "H": 1024, "g": 1280, "G": 1664, "e": 1792}[v],
        "depth": {"Ti": 12, "S": 12, "M": 12, "B": 12, "L": 12, "H": 24, "So400m": 27, "g": 32, "G": 48, "e": 56}[v],
        "mlp_dim": {"Ti": 768, "S": 1536, "M": 2048, "B": 2048, "L": 3072,"So400m": 4304, "H": 4096, "g": 5120, "G": 8192, "e": 15360}[v],
        "num_heads": {"Ti": 3, "S": 6, "M": 8, "B": 8, "L": 12, "So400m": 16, "H": 16, "g": 16, "G": 16, "e": 16}[v],
        # pylint:enable=line-too-long

    }
