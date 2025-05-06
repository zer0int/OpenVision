r'''
this script aims at converting clip_jax model into open_clip compatible model
cd deit_jax; . ~/deit_jax/bv_venv/bin/activate; JAX_PLATFORMS='cpu' python3 -m convert.jax_to_pytorch
copied and modified from E:\研\UCSC\Paper\EfficientCLIP\Code\mine\clip_jax\convert\jax_to_pytorch.py
'''
import functools
import importlib
import inspect
import json
import os
import sys
from pathlib import Path

import open_clip
from absl import flags, app

from ml_collections import config_flags
import numpy as np
import torch
import jax
import jax.numpy as jnp
import orbax
from jax.sharding import Mesh
from flax.linen import partitioning as nn_partitioning
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from huggingface_hub import HfApi, create_repo
from huggingface_hub import login
from huggingface_hub import hf_hub_download

os.environ["XRT_MESH_SERVICE_ADDR"] = ""  # 禁用多主机 mesh 服务
jax.config.update("jax_distributed_debug", True)  # 禁用多主机调试
print("fix addreess")
hf_hub_download = functools.partial(hf_hub_download, library_name="open_clip", library_version='2.26')

SCRIPT_DIR=os.path.realpath(os.path.dirname(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.helpers.sharding import create_mesh, unbox_logicallypartioned, reshard
from src.helpers.utils import load_params, tree_flatten_with_names, recover_dtype, create_orbax_checkpoint_manager
import src.optim.build_optax as build_optax
import open_clip
import src.optim as optim

#########configs
# load config
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")
flags.DEFINE_boolean("cleanup", default=False,
                     help="Delete workdir (only) after successful completion.")
# Adds jax flags to the program.
jax.config.parse_flags_with_absl()
jax.config.update("jax_threefry_partitionable", True)




TXT_MODEL='bert'
TXT_MODEL='vit'
# define some model constants
TRANSPOSE = True # not sure if we need to transpose linear layer weight
ATTENTION_TRANSPOSE = True
# this float32 data type causes small mismatch! like 0.92855495 in LayerNorm scale in jax, becomes 0.9285549521446228027343750000000in pytorch!
NP_DTYPE = np.float32
DTYPE = torch.float32


HF_WEIGHTS_NAME = "open_clip_pytorch_model.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "open_clip_model.safetensors"  # safetensors version
HF_CONFIG_NAME = 'open_clip_config.json'


VISION_MODEL_CONFIG  = \
    {'Ti': {'layers': 12, 'width': 192, 'head_width':64},
     'S': {'layers': 12, 'width': 384, 'head_width':64},
     'B': {'layers': 12, 'width': 768, 'head_width':64},
     'L': {'layers': 24, 'width': 1024, 'head_width':64},
    'So400m': {'layers': 27, 'width': 1152, 'head_width':72,   "mlp_ratio": 3.7362},
     'H': {'layers': 32, 'width': 1280, 'head_width':80},
}

TEXT_MODEL_CONFIG = \
    {'Ti': {'layers': 12, 'width': 192, 'heads': 3},
     'S': {'layers': 12, 'width': 384,  'heads': 6},
     'B': {'layers': 12, 'width': 512,  'heads': 8},
     'L': {'layers': 12, 'width': 768, 'heads':  12},
     'So400m': {'layers': 27, 'width': 1152, 'heads':16, "mlp_ratio": 3.7362},
     'H': {'layers': 24, 'width': 1024, 'heads': 16},
}



##########################

def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32, cls_token=False):
    """Follows the MoCo v3 logic."""
    y, x = jnp.mgrid[:h, :w]

    assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
    omega = jnp.arange(width // 4) / (width // 4 - 1)
    omega = 1. / (temperature ** omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    if cls_token:
        pe = jnp.concatenate([jnp.zeros([1, width]), pe], axis=0)
    # do not need an extra dimension here
    # return jnp.asarray(pe, dtype)[None, :, :]
    return jnp.asarray(pe, dtype)[:, :]


def obrax_to_open_clip(jax_params_cpu, H, W, Channel, pos_embed="learn", txt_model='vit', use_dense_general=True):
    #jax_params_cpu = load_params(None, checkpoint_path)
    #############
    # bfloat16 type gets lost when data is saved to disk, so we recover it.
    #jax_params_cpu = jax.tree_map(recover_dtype, jax_params_cpu)
    # #############
    flat_jax_params_cpu, _ = tree_flatten_with_names(jax_params_cpu)
    flat_jax_params_cpu = {k: v.astype(NP_DTYPE) for k, v in flat_jax_params_cpu}

    torch_state_dict = dict()
    # this set avoids revisiting qkv in a block
    visited_visual_block = set()
    visited_text_block = set()

    if pos_embed == "learn":
        torch_state_dict['visual.positional_embedding'] = torch.tensor(flat_jax_params_cpu['img/pos_embedding'], dtype=DTYPE).squeeze()
        flat_jax_params_cpu.pop('img/pos_embedding')
    elif pos_embed == "sincos2d":
        torch_state_dict['visual.positional_embedding'] = torch.tensor(np.array(posemb_sincos_2d(H, W, Channel, cls_token=True)), dtype=DTYPE)
    else:
        raise ValueError(f"Unknown posemb type: {pos_embed}")

    for k, v in flat_jax_params_cpu.items():
        try:
            # visual ecnoder
            if 'img' in k:
                # transformer block
                if k == 'img/cls':
                    # drop the first two dim of (1,1,1024)
                    torch_state_dict['visual.class_embedding'] = torch.tensor(v[0,0,:], dtype=DTYPE)
                elif k == 'img/embedding/kernel':
                    torch_state_dict['visual.conv1.weight'] = torch.tensor(v.transpose(3, 2, 0, 1), dtype=DTYPE)
                # note that original open_clip does not have conv1.bias
                elif k == 'img/embedding/bias':
                    torch_state_dict['visual.conv1.bias'] = torch.tensor(v, dtype=DTYPE)
                elif k == 'img/encoder_norm/scale':
                    torch_state_dict['visual.ln_post.weight'] = torch.tensor(v, dtype=DTYPE)
                elif k == 'img/encoder_norm/bias':
                    torch_state_dict['visual.ln_post.bias'] = torch.tensor(v, dtype=DTYPE)
                elif k == 'img/head/kernel':
                    torch_state_dict['visual.proj'] = torch.tensor(v, dtype=DTYPE)
                # note that original open_clip does not have proj_bias
                elif k == 'img/head/bias':
                    torch_state_dict['visual.proj_bias'] = torch.tensor(v, dtype=DTYPE)
                elif 'encoderblock_' in k:
                    block_num = int(k.split('_')[1].split('/')[0])
                    if 'LayerNorm_' in k:
                    # a typical key 'img/Transformer/encoderblock_21/LayerNorm_1/bias'
                        norm_num = int(k.split('_')[2].split('/')[0]) + 1 # open_clip layer norm starts from 1
                        if 'bias' in k:
                            torch_state_dict[f'visual.transformer.resblocks.{block_num}.ln_{norm_num}.bias'] = torch.tensor(v, dtype=DTYPE)
                        elif 'scale' in k:
                            torch_state_dict[f'visual.transformer.resblocks.{block_num}.ln_{norm_num}.weight'] = torch.tensor(v, dtype=DTYPE)
                        else:
                            raise ValueError
                    elif 'MlpBlock_0' in k:
                        if 'Dense_0' in k:
                            if 'kernel' in k:
                                weight = v
                                if TRANSPOSE:
                                    weight = weight.transpose(1, 0)
                                torch_state_dict[f'visual.transformer.resblocks.{block_num}.mlp.c_fc.weight'] = torch.tensor(weight, dtype=DTYPE)
                            elif 'bias' in k:
                                torch_state_dict[f'visual.transformer.resblocks.{block_num}.mlp.c_fc.bias'] = torch.tensor(v, dtype=DTYPE)
                            else:
                                raise ValueError
                        elif 'Dense_1' in k:
                            if 'kernel' in k:
                                weight = v
                                if TRANSPOSE:
                                    weight = weight.transpose(1, 0)
                                torch_state_dict[f'visual.transformer.resblocks.{block_num}.mlp.c_proj.weight'] = torch.tensor(weight, dtype=DTYPE)
                            elif 'bias' in k:
                                torch_state_dict[f'visual.transformer.resblocks.{block_num}.mlp.c_proj.bias'] = torch.tensor(v, dtype=DTYPE)
                            else:
                                raise ValueError
                        else:
                            raise ValueError
                    # self attention
                    elif 'MultiHeadDotProductAttention_0' in k:
                        if 'out' in k:
                            if 'bias' in k:
                                torch_state_dict[f'visual.transformer.resblocks.{block_num}.attn.out_proj.bias'] = torch.tensor(v, dtype=DTYPE)
                            else:
                                if use_dense_general:
                                    # shape (16, 64, 1024)
                                    weight = v.reshape(v.shape[0]*v.shape[1], v.shape[2])
                                else:
                                    # nn.dense has a rank of 2
                                    weight = v
                                if TRANSPOSE:
                                    weight = weight.transpose(1, 0)
                                torch_state_dict[f'visual.transformer.resblocks.{block_num}.attn.out_proj.weight'] = torch.tensor(weight, dtype=DTYPE)
                        elif 'query' in k or 'key' in k or 'value' in k:
                            # qkv only needs one visit
                            if block_num in visited_visual_block:
                                continue
                            visited_visual_block.add(block_num)
                            qkv_list = ['query', 'key', 'value']

                            # weight. each is of shape (1024, 16, 64)
                            weight_list = [
                                flat_jax_params_cpu[
                                    f'img/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[0]}/kernel'],
                                flat_jax_params_cpu[
                                    f'img/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[1]}/kernel'],
                                flat_jax_params_cpu[
                                    f'img/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[2]}/kernel'],
                            ]
                            if use_dense_general:
                                weight_list = [weight.reshape(weight.shape[0], weight.shape[1] * weight.shape[2]) for weight in weight_list]
                            if ATTENTION_TRANSPOSE:
                                weight_list = [weight.transpose(1, 0) for weight in weight_list]
                            weight = np.concatenate(weight_list, axis=0)
                            torch_state_dict[f'visual.transformer.resblocks.{block_num}.attn.in_proj_weight'] = torch.tensor(weight, dtype=DTYPE)

                            # bias
                            bias_list = [
                                flat_jax_params_cpu[
                                    f'img/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[0]}/bias'],
                                flat_jax_params_cpu[
                                    f'img/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[1]}/bias'],
                                flat_jax_params_cpu[
                                    f'img/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[2]}/bias'],
                            ]
                            if use_dense_general:
                                bias_list = [bias.reshape(bias.shape[0]*bias.shape[1]) for bias in bias_list]
                            bias = np.concatenate(bias_list, axis=0)
                            torch_state_dict[f'visual.transformer.resblocks.{block_num}.attn.in_proj_bias'] = torch.tensor(bias, dtype=DTYPE)
                        else:
                            raise ValueError
                    else:
                        raise ValueError
                else:
                    raise ValueError
            elif 'txt' in k:
                if txt_model == 'bert':
                    if k == 'txt/head/kernel':
                       # we are using hf model now there is a torch.nn.linear so we have to transpose
                       v = v.transpose(1, 0)
                       torch_state_dict['text.proj.weight'] = torch.tensor(v, dtype=DTYPE)
                    elif 'FlaxBertModule_0' in k:
                        if 'embeddings' in k:
                            if 'LayerNorm' in k:
                                if 'scale' in k:
                                    torch_state_dict['text.transformer.embeddings.LayerNorm.weight'] = torch.tensor(v, dtype=DTYPE)
                                if 'bias' in k:
                                    torch_state_dict['text.transformer.embeddings.LayerNorm.bias'] = torch.tensor(v, dtype=DTYPE)
                            elif 'position_embeddings' in k:
                                torch_state_dict['text.transformer.embeddings.position_embeddings.weight'] = torch.tensor(v, dtype=DTYPE)
                            elif 'token_type_embeddings' in k:
                                torch_state_dict['text.transformer.embeddings.token_type_embeddings.weight'] = torch.tensor(v, dtype=DTYPE)
                            elif 'word_embeddings' in k:
                                torch_state_dict['text.transformer.embeddings.word_embeddings.weight'] = torch.tensor(v, dtype=DTYPE)

                        elif 'encoder' in k:
                            if 'kernel' in k:
                                jax_weight =  torch.tensor(v.transpose(1, 0))
                            else:
                                jax_weight =  torch.tensor(v, dtype=DTYPE)
                            layer_id = int(k.split('/')[4])
                            print(f'We are converting layer {layer_id}')
                            if 'attention' in k:
                                if 'self' in k:
                                    if 'query' in k:
                                        if 'kernel' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.self.query.weight'] = jax_weight
                                        elif 'bias' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.self.query.bias'] = jax_weight
                                    elif 'key' in k:
                                        if 'kernel' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.self.key.weight'] = jax_weight
                                        elif 'bias' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.self.key.bias'] = jax_weight
                                    elif 'value' in k:
                                        if 'kernel' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.self.value.weight'] = jax_weight
                                        elif 'bias' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.self.value.bias'] = jax_weight
                                else:
                                    if 'output/dense' in k:
                                        if 'kernel' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.output.dense.weight'] = jax_weight
                                        elif 'bias' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.output.dense.bias'] = jax_weight
                                    elif 'output/LayerNorm' in k:
                                        if 'scale' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.output.LayerNorm.weight'] = jax_weight
                                        elif 'bias' in k:
                                            torch_state_dict[
                                                f'text.transformer.encoder.layer.{layer_id}.attention.output.LayerNorm.bias'] = jax_weight

                            # 判断是否为中间层(intermediate)
                            elif 'intermediate' in k:
                                if 'kernel' in k:
                                    torch_state_dict[
                                        f'text.transformer.encoder.layer.{layer_id}.intermediate.dense.weight'] = jax_weight
                                elif 'bias' in k:
                                    torch_state_dict[
                                        f'text.transformer.encoder.layer.{layer_id}.intermediate.dense.bias'] = jax_weight

                            # 判断是否为输出层(output)
                            else:
                                if 'output/dense' in k:
                                    if 'kernel' in k:
                                        torch_state_dict[
                                            f'text.transformer.encoder.layer.{layer_id}.output.dense.weight'] = jax_weight
                                    elif 'bias' in k:
                                        torch_state_dict[
                                            f'text.transformer.encoder.layer.{layer_id}.output.dense.bias'] = jax_weight
                                elif 'output/LayerNorm' in k:
                                    if 'scale' in k:
                                        torch_state_dict[
                                            f'text.transformer.encoder.layer.{layer_id}.output.LayerNorm.weight'] = jax_weight
                                    elif 'bias' in k:
                                        torch_state_dict[
                                            f'text.transformer.encoder.layer.{layer_id}.output.LayerNorm.bias'] = jax_weight

                elif txt_model== 'vit':
                    if k == 'txt/pos_embedding':
                        # (1, 32, 768) -> (32, 768)
                        torch_state_dict['positional_embedding'] = torch.tensor(v[0], dtype=DTYPE)
                    elif k == 'txt/Embed_0/embedding':
                        print(f'token_embedding has a shape of {v.shape}')
                        torch_state_dict['token_embedding.weight'] = torch.tensor(v, dtype=DTYPE)
                    elif k == 'txt/encoder_norm/scale':
                        torch_state_dict['ln_final.weight'] = torch.tensor(v, dtype=DTYPE)
                    elif k == 'txt/encoder_norm/bias':
                        torch_state_dict['ln_final.bias'] = torch.tensor(v, dtype=DTYPE)
                    elif k == 'txt/head/kernel':
                        torch_state_dict['text_projection'] = torch.tensor(v, dtype=DTYPE)
                    elif 'encoderblock_' in k:
                        block_num = int(k.split('_')[1].split('/')[0])
                        if 'LayerNorm_' in k:
                            # a typical key 'txt/Transformer/encoderblock_8/LayerNorm_0/scale'
                            norm_num = int(k.split('_')[2].split('/')[0]) + 1  # open_clip layer norm starts from 1
                            if 'bias' in k:
                                torch_state_dict[f'transformer.resblocks.{block_num}.ln_{norm_num}.bias'] = torch.tensor(v, dtype=DTYPE)
                            elif 'scale' in k:
                                torch_state_dict[f'transformer.resblocks.{block_num}.ln_{norm_num}.weight'] = torch.tensor(v, dtype=DTYPE)
                            else:
                                raise ValueError
                        elif 'MlpBlock_0' in k:
                            # a typical key 'txt/Transformer/encoderblock_8/MlpBlock_0/Dense_0/kernel'
                            if 'Dense_0' in k:
                                if 'kernel' in k:
                                    weight = v
                                    if TRANSPOSE:
                                        weight = weight.transpose(1, 0)
                                    torch_state_dict[f'transformer.resblocks.{block_num}.mlp.c_fc.weight'] = torch.tensor(weight, dtype=DTYPE)
                                elif 'bias' in k:
                                    torch_state_dict[f'transformer.resblocks.{block_num}.mlp.c_fc.bias'] = torch.tensor(v, dtype=DTYPE)
                                else:
                                    raise ValueError
                            elif 'Dense_1' in k:
                                if 'kernel' in k:
                                    weight = v
                                    if TRANSPOSE:
                                        weight = weight.transpose(1, 0)
                                    torch_state_dict[f'transformer.resblocks.{block_num}.mlp.c_proj.weight'] = torch.tensor(weight, dtype=DTYPE)
                                elif 'bias' in k:
                                    torch_state_dict[f'transformer.resblocks.{block_num}.mlp.c_proj.bias'] = torch.tensor(v, dtype=DTYPE)
                                else:
                                    raise ValueError
                        elif 'MultiHeadDotProductAttention_0' in k:
                            if 'out' in k:
                                if 'bias' in k:
                                    torch_state_dict[f'transformer.resblocks.{block_num}.attn.out_proj.bias'] = torch.tensor(v, dtype=DTYPE)
                                elif 'kernel' in k:
                                    if use_dense_general:
                                        weight = v.reshape(v.shape[0]*v.shape[1], v.shape[2])
                                    else:
                                        # nn.dense has a rank of 2
                                        weight = v
                                    if TRANSPOSE:
                                        weight = weight.transpose(1, 0)
                                    torch_state_dict[f'transformer.resblocks.{block_num}.attn.out_proj.weight'] = torch.tensor(weight, dtype=DTYPE)
                                else:
                                    raise NotImplementedError
                            elif 'query' in k or 'key' in k or 'value' in k:
                                if block_num in visited_text_block:
                                    continue
                                visited_text_block.add(block_num)
                                qkv_list = ['query', 'key', 'value']

                                weight_list = [
                                    flat_jax_params_cpu[
                                        f'txt/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[0]}/kernel'],
                                    flat_jax_params_cpu[
                                        f'txt/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[1]}/kernel'],
                                    flat_jax_params_cpu[
                                        f'txt/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[2]}/kernel'],
                                ]
                                if use_dense_general:
                                    weight_list = [weight.reshape(weight.shape[0], weight.shape[1] * weight.shape[2]) for weight in weight_list]
                                if ATTENTION_TRANSPOSE:
                                    weight_list = [weight.transpose(1, 0) for weight in weight_list]
                                weight = np.concatenate(weight_list, axis=0)
                                torch_state_dict[f'transformer.resblocks.{block_num}.attn.in_proj_weight'] = torch.tensor(weight, dtype=DTYPE)

                                bias_list = [
                                    flat_jax_params_cpu[
                                        f'txt/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[0]}/bias'],
                                    flat_jax_params_cpu[
                                        f'txt/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[1]}/bias'],
                                    flat_jax_params_cpu[
                                        f'txt/Transformer/encoderblock_{block_num}/MultiHeadDotProductAttention_0/{qkv_list[2]}/bias'],
                                ]
                                if use_dense_general:
                                    bias_list = [bias.reshape(bias.shape[0]*bias.shape[1]) for bias in bias_list]
                                bias = np.concatenate(bias_list, axis=0)
                                torch_state_dict[f'transformer.resblocks.{block_num}.attn.in_proj_bias'] = torch.tensor(bias, dtype=DTYPE)
                            else:
                                raise ValueError
                        else:
                            raise ValueError
                else:
                    raise ValueError
            elif k == 't':
                torch_state_dict['logit_scale'] = torch.tensor(flat_jax_params_cpu['t'][0])
            else:
                raise ValueError
        except:
            # FIXME change back!
            import pdb
            # pdb.set_trace()

    #torch.save(torch_state_dict, new_checkpoint_path)
    print(f'old jax checkpoint  converted finihsed')
    return torch_state_dict



def load_obrax_ckpt(argv):

    del argv

    config = flags.FLAGS.config
    work_dir = flags.FLAGS.workdir
    device_arrays = create_mesh(config)
    mesh = Mesh(device_arrays, config.sharding.mesh_axes)
    repl_sharding  = jax.sharding.NamedSharding(mesh, P())

    rng = jax.random.PRNGKey(jax.device_put(config.get("seed", 0),
                                            jax.local_devices(backend="cpu")[0]))

    model_mod = importlib.import_module(f"src.models.{config.model_name}")
    model = model_mod.Model(**config.get("model", {}), mesh=mesh)

    # here we only use init function to have abstract parameter state
    def init(rng):
        image_size = config.init_shapes[0]
        text_size = config.init_shapes[1]
        no_image = jnp.zeros(image_size, jnp.float32)
        no_text = jnp.zeros(text_size, jnp.int32)
        params = model.init(rng, no_image, no_text, train=True)["params"]
        return params

    # parameters related shape/array/sharding
    with nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        params_shape = jax.eval_shape(init, rng)

    params_logical_annotations = nn.get_partition_spec(params_shape)

    params_mesh_shardings = nn.logical_to_mesh_sharding(params_logical_annotations, mesh,
                                                        config.sharding.logical_axis_rules)
    params_unboxed_shape = unbox_logicallypartioned(params_shape)


    # optimizer related
    tx, _ = optim.make(config, params_unboxed_shape, sched_kw=dict(
        total_steps=config.total_steps, batch_size=config.input.batch_size, data_size=1000000000)) # use dummy value since we don't need opt

    # opt state sharding
    with nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        opt_shape = jax.eval_shape(tx.init, params_unboxed_shape)
    opt_logical_annotations = nn.get_partition_spec(opt_shape)

    opt_mesh_shardings = nn.logical_to_mesh_sharding(opt_logical_annotations,
                                                     mesh,
                                                     config.sharding.logical_axis_rules)
    rng = reshard(rng, repl_sharding)

    # init params and opt
    params = jax.jit(init, in_shardings=None, out_shardings=params_mesh_shardings)(rng)
    opt = jax.jit(tx.init, out_shardings=opt_mesh_shardings)(params)

    params = unbox_logicallypartioned(params)
    opt = unbox_logicallypartioned(opt)
    train_state = {"params": params, "opt": opt}
    del params, opt  # Delete to avoid memory leak or accidental reuse.

    workdir = flags.FLAGS.workdir
    save_ckpt_path = os.path.join(workdir, "checkpoint.npz")
    abstract_train_state = jax.tree_util.tree_map(
        orbax.checkpoint.utils.to_shape_dtype_struct, train_state
    )
    ckpt_mngr  = create_orbax_checkpoint_manager(
                save_ckpt_path,
                True,
                True,
                save_interval_steps=1,
                max_to_keep=1,
            )
    latest_step = ckpt_mngr.latest_step()
    train_state = ckpt_mngr.restore(
        latest_step,
        args=orbax.checkpoint.args.StandardRestore(abstract_train_state),
    )
    print('loading successfully')
    params = jax.device_get(train_state['params'])
    torch_params = obrax_to_open_clip(params,
                       H= config.init_shapes[0][1]//int(config.model.image.variant.split('/')[-1]),
                       W= config.init_shapes[0][1]//int(config.model.image.variant.split('/')[-1]),
                       Channel=int( VISION_MODEL_CONFIG[config.model.image.variant.split('/')[0]]['width']), # this is vision positional embedding
                       pos_embed=config.model.image.posemb,
                       txt_model=TXT_MODEL,
                       use_dense_general=config.model.image.use_dense_general)

    # upload to HF
    save_and_upload_for_hf(
        torch_params,
        config,
        text_model=TXT_MODEL
    )
    # verify the openclip weights and jax weights
    openclip_model, _, _ = open_clip.create_model_and_transforms(f'hf-hub:{config.hf_upload.repo_name}')
    openclip_model.eval() # bert model has 0.1 dropout in output layer
    torch_output = openclip_model(torch.ones((1, 3, config.init_shapes[0][1], config.init_shapes[0][1])), torch.ones((1, config.init_shapes[1][1])).long())

    @functools.partial(jax.jit, backend='cpu')
    def jax_model_inference(params, image, text):
        return model.apply({"params": params},
                      image, text)
    jax_output = jax_model_inference(params, jnp.ones((1, config.init_shapes[0][1], config.init_shapes[0][1], 3)), jnp.ones((1, config.init_shapes[1][1])))
    print('image_feature is gap is ', torch_output[0].detach().numpy().mean() - np.array(jax_output[0]).mean())
    print('text_feature is gap is ', torch_output[1].detach().numpy().mean() - np.array(jax_output[1]).mean())



def _get_hf_config_tokenizer(model_id, cache_dir=None):
    config_path  = hf_hub_download(model_id,  filename='open_clip_config.json', cache_dir=cache_dir, revision=None)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    tokenizer = open_clip.get_tokenizer(f'hf-hub:{model_id}')
    return config, tokenizer


def save_and_upload_for_hf(
    model_state_dict,
    bv_conifg,
    text_model = 'vit'
):
    config_filename = HF_CONFIG_NAME
    #
    model_config, tokenizer = _get_hf_config_tokenizer(bv_conifg.hf_upload.model_id,
                                                       cache_dir=bv_conifg.hf_upload.cache_dir)

    # first revise text model config
    if text_model == 'bert':
        biomed_model_config, tokenizer = _get_hf_config_tokenizer(bv_conifg.openclip_tokenizer.repo_name,
                                                           cache_dir=bv_conifg.hf_upload.cache_dir)
        model_config['model_cfg']['text_cfg'] = biomed_model_config['model_cfg']['text_cfg']
        # hack we currently only use single dense layer
        model_config['model_cfg']['text_cfg']['hf_proj_type'] = 'linear'
    else:
        # use vit's config to updated
        model_config['model_cfg']['text_cfg'].update(
            TEXT_MODEL_CONFIG[bv_conifg.model.text.variant.split('/')[0]]
        )
        # hack
        model_config['model_cfg']['text_cfg']['act_kwargs'] = {'approximate':'tanh'}

        model_config['model_cfg']['text_cfg']['pool_type'] = {'tok': 'first',
                                                              'last': 'last'}[bv_conifg.model.text.pool_type]
    model_config['model_cfg']['text_cfg']['context_length'] = bv_conifg.input.txt_token_length

    # then update tokenizer information if needed
    if bv_conifg.get('openclip_tokenizer.enable', False):
        print(f'We are saving the new openclip tokenizer from {bv_conifg.openclip_tokenizer.repo_name}')
        tokenizer = open_clip.get_tokenizer(f'hf-hub:{bv_conifg.openclip_tokenizer.repo_name}')

        model_config['model_cfg']['text_cfg']['vocab_size'] =  int(bv_conifg.model.text.vocab_size)

        model_config['model_cfg']['text_cfg']['hf_tokenizer_name'] = bv_conifg.openclip_tokenizer.repo_name


    ### revise vision config here
    model_config['model_cfg']['embed_dim'] = int(bv_conifg.model.out_dim[0])
    model_config['model_cfg']['vision_cfg'].update(
        VISION_MODEL_CONFIG[bv_conifg.model.image.variant.split('/')[0]] | {
            'patch_size': int(bv_conifg.model.image.variant.split('/')[1])}
    )

    model_config['model_cfg']['vision_cfg']['image_size'] = bv_conifg.init_shapes[0][1]
    #save model and configs
    save_directory = Path(bv_conifg.hf_upload.save_directory)
    save_directory.mkdir(exist_ok=True, parents=True)

    # save jax weights in the save_directory
    # call the gcs api to download the weights
    # from gcs to local
    save_ckpt_path = os.path.join(flags.FLAGS.workdir, "checkpoint.npz")
    jax_weight_dir = os.path.join(save_directory, "jax_orbax_weight")
    os.makedirs(jax_weight_dir, exist_ok=True)
    
    # use gsutil command to copy weight file
    gsutil_cmd = f"gsutil cp -r {save_ckpt_path} {jax_weight_dir}/"
    print(f"Executing command: {gsutil_cmd}")
    os.system(gsutil_cmd)
    print(f"JAX weights have been downloaded to: {jax_weight_dir}")


    torch.save(model_state_dict, save_directory / HF_WEIGHTS_NAME)
    tokenizer.save_pretrained(save_directory)
    # save new config
    config_path = save_directory / config_filename
    with config_path.open('w') as f:
        json.dump(model_config, f, indent=2)

    # upload to hf
    login(token=bv_conifg.hf_upload.token)
    api = HfApi()
    create_repo(bv_conifg.hf_upload.repo_name, private=True, repo_type='model', exist_ok=True)
    api.upload_folder(
        folder_path=save_directory,
        repo_id=bv_conifg.hf_upload.repo_name,
        commit_message=bv_conifg.hf_upload.commit_message,
        path_in_repo="",
    )
    print('finish uploading')






if __name__ == '__main__':
    # load orbax weight and transfer to CPU
    app.run(load_obrax_ckpt)
