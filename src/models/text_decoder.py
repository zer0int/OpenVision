# #Copyright @2024 Zeyu Wang
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

# 标准库导入
import functools
import sys
from typing import Any, Callable, Optional, Sequence, Tuple, Union


from absl import logging
from einops import rearrange, repeat
import flax
import flax.linen as nn
import flax.training.checkpoints
from flax.linen.linear import DenseGeneral
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


from src.helpers import utils
from src.models import common
from src.models.bpt import blockwise_attn
from src.models.common import DropPath
from src.models.text_transformer import Encoder, MlpBlock, Encoder1DBlock

Array = Any
Dtype = Any  # this could be a real type?

class MultiHeadDotProductAttention(nn.MultiHeadDotProductAttention):

    attn_kernel_init: Callable = nn.initializers.normal(stddev=0.01)
    proj_kernel_init: Callable = nn.initializers.normal(stddev=0.01)
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Any  = None
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

        dense = functools.partial(DenseGeneral,
                                  axis=-1,
                                  dtype=self.dtype,
                                  param_dtype=self.param_dtype,
                                  features=(self.num_heads, head_dim),
                                  kernel_init=nn.with_logical_partitioning(self.attn_kernel_init, ("embed", "heads")),
                                  bias_init=nn.with_logical_partitioning(self.bias_init, (None,)),
                                  use_bias=self.use_bias,
                                  precision=self.precision)

        query, key, value = (dense(name='query')(inputs_q),
                             dense(name='key')(inputs_kv),
                             dense(name='value')(inputs_kv))

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

        if not self.use_flash_attn:
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
                query, key, value, decoder_segment_ids=None
            )
        x = nn.with_logical_constraint(x,
                                       ("activation_batch", "activation_length", "activation_heads", "activation_kv"))

        x = DenseGeneral(features=features,
                           axis=(-2, -1),
                           kernel_init=nn.with_logical_partitioning(self.proj_kernel_init, ("heads", "embed"),),
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

    def _tpu_flash_attention(self, query: Array, key: Array, value: Array, decoder_segment_ids: Array | None) -> Array:
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
        def wrap_flash_attention(query, key, value, decoder_segment_ids):
            if decoder_segment_ids is not None:
                assert (
                        query.shape[2]
                        == decoder_segment_ids.q.shape[1]
                ), 'Sharding along sequence dimension not allowed in flash attention'

            return tpu_flash_attention.flash_attention(
                query,
                key,
                value,
                causal=False,
                segment_ids=decoder_segment_ids,
                sm_scale=64 ** -0.5,
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
        x = wrap_flash_attention(query, key, value, decoder_segment_ids)
        x = jnp.transpose(x, axes=(0, 2, 1, 3))
        return x


class CrossAttnEncoder1DBlock(nn.Module):
    """Single transformer encoder block (MHSA + MLP)."""
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    drop_path: float = 0.0
    depth: int = 12
    casual_mask: bool = False
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Any = None

    @nn.compact
    def __call__(self, x, u, attn_mask=None, deterministic=True):
        width = x.shape[-1]
        init_std = {
            'proj': (width ** -0.5) * ((2 * self.depth) ** -0.5),
            'attn': width ** -0.5,
            'fc': (2 * width) ** -0.5
        }
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

        u = u.astype(self.dtype)
        u = nn.with_logical_constraint(u, ("activation_batch", "activation_length", "activation_embed"))
        v = nn.LayerNorm(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("norm",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), (None,)),
           )(u)
        v = nn.with_logical_constraint(v, ("activation_batch", "activation_length", "activation_embed"))
        
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
        )(y, v, mask=attn_mask if self.casual_mask else None)
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        # y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
        x = out["+sa"] = x + y

        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))

        y = nn.LayerNorm(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            scale_init=nn.with_logical_partitioning(nn.initializers.ones_init(), ("norm",)),
            bias_init=nn.with_logical_partitioning(nn.initializers.zeros_init(), (None,)),
        )(x)
        y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim, dropout=self.dropout,
            fc_init=nn.initializers.normal(stddev=init_std['fc']),
            proj_init=nn.initializers.normal(stddev=init_std['proj']),
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(y, deterministic)
        y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
        y = nn.with_logical_constraint(y, ("activation_batch", "activation_length", "activation_embed"))
        # y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
        x = out["+mlp"] = x + y
        x = nn.with_logical_constraint(x, ("activation_batch", "activation_length", "activation_embed"))
        return x, out

from flax.linen.partitioning import remat

class CrossAttnEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""
    depth: int
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    # drop_path: float = 0.0
    remat_policy: str = "none"
    drop_path: float = 0.0
    casual_mask: bool = False
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Any = None

    @nn.compact
    def __call__(self, x, u, deterministic=True):
        out = {}
        dpr = [
            float(x) for x in np.linspace(
                0,
                self.drop_path,
                self.depth)]  # drop path decay
        # Input Encoder
        CrossAttnBlockLayer = CrossAttnEncoder1DBlock
        if self.remat_policy not in (None, "none"):
            logging.info(f"remat policy: {self.remat_policy}")
            if self.remat_policy == "minimal":
                policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
            else:
                policy = None
            logging.info(f"activation checkpointing {self.remat_policy}")
            CrossAttnBlockLayer = remat(  # pylint: disable=invalid-name
                CrossAttnEncoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(3,)
            )  # "deterministic" is a static argument in CrossAttnEncoder1DBlock

        BlockLayer = Encoder1DBlock
        if self.remat_policy not in (None, "none"):
            logging.info(f"remat policy: {self.remat_policy}")
            if self.remat_policy == "minimal":
                policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
            else:
                policy = None
            logging.info(f"activation checkpointing {self.remat_policy}")
            BlockLayer = remat(  # pylint: disable=invalid-name
                Encoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(1,)
            )  # "deterministic" is a static argument in Encoder1DBlock

        for lyr in range(self.depth):
            x, out[f"block{lyr:02d}"] = BlockLayer(
                name=f"encoderblock_{lyr}",
                mlp_dim=self.mlp_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                dropout=self.dropout,
                drop_path=dpr[lyr],
                casual_mask=self.casual_mask,
                use_flash_attn=self.use_flash_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                mesh=self.mesh
            )(x, deterministic)
            x, out[f"crossattn_block{lyr:02d}"] = CrossAttnBlockLayer(
                name=f"crossattn_encoderblock_{lyr}",
                mlp_dim=self.mlp_dim, 
                depth=self.depth,
                num_heads=self.num_heads, 
                dropout=self.dropout,
                drop_path=dpr[lyr],
                casual_mask=False,
                use_flash_attn=self.use_flash_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                mesh=self.mesh)(x, u, None, deterministic)
        out["pre_ln"] = x

        return x, out


class _Model(nn.Module):
    """A image-text autoregression Transformer model."""
    num_classes: int = None
    width: int = 512
    depth: int = 12
    mlp_dim: Optional[int] = None  # Defaults to 4x input dim
    num_heads: int = 12
    dropout: float = 0.0
    remat_policy: str = 'none'
    fusion_style: str = 'cross_attn'
    scan_mlp: bool = False
    scan_attn: bool = False
    mlp_chunck: int = 128
    casual_mask: bool = True
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Any = None
    drop_path: float = 0.0
    num_learnable_tokens: int = 80
    drop_token: int = 0

    @nn.compact
    def __call__(self, image_embeds, text_embeds, *, train=False):
        out = {}

        if self.drop_token > 0:
            image_embeds = image_embeds[:, :image_embeds.shape[1] - self.drop_token + 1]

        ni, li, di = image_embeds.shape
        nt, lt, dt = text_embeds.shape
        assert ni == nt, f'the image embed is {image_embeds.shape} and text embed is {text_embeds.shape}'

        image_projection_layer = nn.Dense(
            self.width,
            name="image_projection_layer",
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=di ** -0.5))
        image_embeds = out["projected_image_embeds"] = image_projection_layer(image_embeds)

        text_projection_layer = nn.Dense(
            self.width,
            name="text_projection_layer",
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=dt ** -0.5))
        text_embeds = out["projected_text_embeds"] = text_projection_layer(text_embeds)
        
        learnable_tokens = self.param(
            'learnable_tokens',
            nn.initializers.normal(stddev=1.0),
            (self.num_learnable_tokens, self.width)
        )
        learnable_tokens = jnp.tile(learnable_tokens[None, :, :], (ni, 1, 1))

        image_embeds = nn.with_logical_constraint(image_embeds, ("activation_batch", "activation_length", "activation_embed"))
        text_embeds = nn.with_logical_constraint(text_embeds, ("activation_batch", "activation_length", "activation_embed"))

        # concatenate image_embeds and learnable_tokens in token dimension
        image_embeds = jnp.concatenate([image_embeds, text_embeds], axis=1)
        li = image_embeds.shape[1]  # update image_embeds token dimension

        # keep text_embeds unchanged or continue processing
        # if you need to further process the text embeds, you can add logic here
        # currently, text_embeds will remain unchanged
        text_embeds = learnable_tokens
        lt = text_embeds.shape[1]

        # TODO: figure out if need to do triu
        if self.fusion_style == 'concat':

            image_text_embeds = jnp.concatenate((image_embeds, text_embeds), axis=1)
            image_text_embeds = nn.with_logical_constraint(image_text_embeds,
                                                     ("activation_batch", "activation_length", "activation_embed"))

            # it is actually decoder, but we are importing text_transformer implementation to save effort
            decoder_blocks = Encoder(
                depth=self.depth,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                drop_path=self.drop_path,
                remat_policy=self.remat_policy,
                casual_mask=self.casual_mask,
                use_flash_attn=self.use_flash_attn,
                scan_mlp=self.scan_mlp,
                scan_attn=self.scan_attn,
                mlp_chunck=self.mlp_chunck,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                mesh=self.mesh,
                fusion_style=self.fusion_style,
                li=li,
                lt=lt,
                name="Transformer")

            x, out["decoder"] = decoder_blocks(
                image_text_embeds, deterministic=not train)

            x = x[:, li:]
            def truncate_img_tokens(elem):
                for k, v in elem.items():
                    if isinstance(v, dict):
                        elem[k] = truncate_img_tokens(v)
                    else:
                        elem[k] = v[:, li:]
                return elem
            out["decoder"] = truncate_img_tokens(out["decoder"])
        elif self.fusion_style == 'cross_attn':
            # assume there is a <s> at the beginning of text
            # no need to attn_pool vit feature like coca
            # qformer-like arch is shown to be less effective. https://www.zhihu.com/question/626796690/answer/3532414535
            # we do not do attention pool as this is mostly aligned with llava practice

            # has to use causal mask to avoid trival learning
            # follow
            # https://github.com/google/jax/discussions/19905
            # https://github.com/google/maxtext/blob/e9e9e1786d2839c9c4b3dc35c26f25535f616e38/MaxText/layers/attentions.py#L168
            # but different from pytorch implementation
            # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/transformer.py#L879
            # https://github.com/huggingface/transformers/blob/6af0854efa3693e0b38c936707966685ec3d0ae8/src/transformers/models/llama/modeling_llama.py#L96
            # attn_mask = jnp.tri(lt, dtype=bool)
            # attn_mask = repeat(attn_mask, 'l m -> b 1 l m', b=nt)
            # we can also make cls_token not attending to [PAD} tokens, but that won't be necessary

            # it is actually decoder, but we are inheriting text_transformer naming for legacy consistency
            assert self.depth % 2 == 0
            decoder_blocks = CrossAttnEncoder(
                depth=self.depth // 2,
                mlp_dim=self.mlp_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                drop_path=self.drop_path,
                remat_policy=self.remat_policy,
                casual_mask=self.casual_mask,
                use_flash_attn=self.use_flash_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                mesh=self.mesh,
                name="Transformer")

            x, out["decoder"] = decoder_blocks(
                text_embeds, image_embeds, deterministic=not train)
        else:
            raise ValueError

        x = out["norm"] = nn.LayerNorm(name="decoder_norm")(x)
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

        return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
    """Factory function, because linen really don't like what I'm doing!"""
    return _Model(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
    """Converts a string like "B" or "B/32" into a params dict."""
    if variant is None:
        return {}

    v = variant

    return {
        # pylint:disable=line-too-long
        # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
        # from text transformer
        "width": {"Ti": 192, "S": 384, "M": 512, "B": 512, "L": 768, "So400m": 1152, "H": 1024, "g": 1024, "G": 1664, "e": 1792}[v],
        "depth": {"Ti": 12, "S": 12, "M": 12, "B": 12, "L": 12, "So400m": 27,"H": 24, "g": 24, "G": 48, "e": 56}[v],
        "mlp_dim": {"Ti": 768, "S": 1536, "M": 2048, "B": 2048, "L": 3072, "So400m": 4304,"H": 4096, "g": 4096, "G": 8192, "e": 15360}[v],
        "num_heads": {"Ti": 3, "S": 6, "M": 8, "B": 8, "L": 12, "So400m": 16, "H": 16, "g": 16, "G": 16, "e": 16}[v],
        # pylint:enable=line-too-long
    }


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  del model_cfg

  restored_params = utils.load_params(None, init_file)

  # possibly use the random init for some of the params (such as, the head).
  restored_params = common.merge_params(restored_params, init_params, dont_load=dont_load)

  from helpers.utils import recover_dtype
  restored_params =  jax.tree_map(recover_dtype, restored_params)
  return restored_params