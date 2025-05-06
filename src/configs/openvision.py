# Copyright 2022 Big Vision Authors.
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

# pylint: disable=line-too-long

import functools

import jax.numpy

import src.configs.common as bvcc
import src.configs.clip_common as common
from ml_collections import ConfigDict


def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg,
      res=112,
      batch_factor=2.,
      base_lr=8e-6,
      txt_key='llava_caption',
      imagenet_epoch=2000,
      vitual_warmup_epoch=20,
      runlocal=False,
      token_len=80,
      output_token_len=128,
      concat=False,
      prob=1.0,
      remat='full',
      txt='bert_base',
      img='L/16',
      txt_name='L/16',
      init='',
      data_parallelism=128,
      fsdp_parallelism=2,
      tensor_parallelism=1,
      img_head=True,
      load_pretrain=False,
      use_sovit=False,
      use_openclip_tokenizer=False,
      mask_ratio=0.,
      txt_key1='txt',
      txt_key2='llava_caption',
      color_jitter=True,
      vocab_path='./bert_base_vocab_bos_eos.txt',
      txt_decoder_name='L', vocab_size=32000)

  config = ConfigDict()

  #####################################
  #            sharding               #
  #####################################
  config.sharding = dict()
  config.sharding.meshshape = dict(
      data_parallelism=arg.data_parallelism,
      fsdp_parallelism=arg.fsdp_parallelism,
      tensor_parallelism=arg.tensor_parallelism,
  )
  config.sharding.mesh_axes = ['data', 'fsdp', 'tensor']
  config.sharding.data_sharding = [['data', 'fsdp', 'tensor']]

  config.sharding.logical_axis_rules = [
      ['activation_batch', ['data', 'fsdp']],
      ['activation_heads', ['tensor']],
      ['activation_length', []],
      ['activation_embed', ['tensor']],
      ['activation_mlp', ['tensor']],
      ['activation_kv', ['tensor']],
      ['activation_vocab', ['tensor']],
      ['activation_vocab', []],
      ['mlp', 'tensor'],
      ['vocab', 'tensor'],
      ['embed', 'fsdp'],
      ['norm', 'tensor'],
      ['heads', 'tensor'],
      ['kv', []],
  ]

  # wandb config
  config.wandb = dict(
      log_wandb=False,
      wandb_offline=False,
      resume=False,
      debug_data=False,
      project='your wandb project',
      experiment=f'your wandb experiment',
      entity='your wandb entity'
  )

  # ckpt config
  config.save_ckpt = True
  config.keep_ckpt = 100000000
  config.ckpt_steps = 1000
  config.log_training_steps = 50

  # input config
  config.input = {}
  config.input.data = dict(name='tfds dataset name',
                           split='your split',
                           data_dir='your data dir')
  config.input.cach_raw = True
  config.input.shuffle_buffer_size = 250_000  if not arg.runlocal else 50
  config.input.txt_token_length = arg.token_len
  config.init_shapes = [(128, arg.res, arg.res, 3), (256, arg.token_len,)]
  config.init_types = ['float32', 'int32']


  vocab_path = arg.vocab_path
  if arg.use_openclip_tokenizer:
      text_pp =  f'|flatten|copy("{arg.txt_key1}", "labels")|keep("image", "labels")'
  else:
      tokenizer = f'my_bert_tokenize(max_len={arg.token_len}, output_token_len={arg.output_token_len}, vocab_path="{vocab_path}", add_bos=True, add_eos=True, key1="{arg.txt_key1}", key2="{arg.txt_key2}")'
      text_pp =  f'|flatten|{tokenizer}|get_autoreg_label(pad_token=0)|keep("image", "labels1", "labels2", "autoreg_labels", "cap_loss_mask")'

  if arg.color_jitter:
      input_pp =  f'inception_crop(inkey="jpg", size={arg.res}, area_min=40, method="bilinear", antialias=True)|simclr_jitter_gray(jitter_strength=0.4)'
  else:
      input_pp =  f'inception_crop(inkey="jpg", size={arg.res}, area_min=40, method="bilinear", antialias=True)'

  config.input.pp  = (
      input_pp+text_pp

  )
  config.pp_modules = [
      'ops_general', 'ops_image', 'ops_text', 'bert_ops']



  # Model section
  config.model_name = 'two_towers'
  config.model_load = {}

  # load config if you want to load a pretrained model
  config.load_config = ConfigDict()
  config.load_config.model = ConfigDict(
      dict(
          image_model='vit',
          text_model='text_transformer',
          text_decoder='text_decoder',
          text_decoder_config= ConfigDict({
          'variant': 'B',
          'num_classes': arg.vocab_size,
          'dtype': 'float32',
          'param_dtype': 'float32',
          'remat_policy': 'none',
          'fusion_style': 'concat',
          'casual_mask': True,
          'num_learnable_tokens':arg.output_token_len
      }),
          image=ConfigDict({
              'variant': 'B/16',
              'posemb': 'sincos2d',
              'remat_policy': arg.remat,
              'mask_ratio': arg.mask_ratio,
              'use_flash_attn': False,
              'emb_head_bias': False,
              'head_zeroinit': False,
              'dtype': 'float32',
              'param_dtype': 'float32',
              'output_tokens': True,
              'use_dense_general': False,
              'pool_type': 'gap'
          }),
          text=ConfigDict({
              'variant': 'B/16',
              'pool_type': 'last',
              'use_flash_attn': False,
              'remat_policy': arg.remat,
              'casual_mask': False,
              'head_zeroinit': False,
              'dtype': 'float32',
              'param_dtype': 'float32',
              'vocab_size': arg.vocab_size,
              "embed_cls": True,
              "output_tokens": True,
              'use_dense_general': False
          }),
          temperature_init=1 / 0.07,
          out_dim=512,

      )
  )

  config.load_config.transform = ConfigDict(
      {'patch': False,
      'patch_init': 'interp'})

  config.model = ConfigDict()
  config.model.image_model = 'vit'
  config.model.text_model = 'text_transformer'
  config.model.text_decoder = 'text_decoder'
  config.model.text_decoder_config = ConfigDict({
      'variant': arg.txt_decoder_name,
      'num_classes': arg.vocab_size,
      'dtype': 'float32',
      'scan_mlp': False,
      'scan_attn': False,
      'use_flash_attn': False,
      'mlp_chunck':  128,
      'param_dtype': 'float32',
      'remat_policy': 'none',
      'fusion_style': 'concat',
      'casual_mask': True,
      'num_learnable_tokens':arg.output_token_len,
      'drop_token': 0,
  })
  config.model.image = ConfigDict({
      'variant': arg.img,
      'posemb': 'sincos2d',
      'scan_mlp': False,
      'scan_attn': False,
      'mlp_chunck': 128,
      'ignore_cls': False,
      'remat_policy':arg.remat,
      'mask_ratio': arg.mask_ratio,
      'use_flash_attn': False,
      'emb_head_bias':False,
      'head_zeroinit': False,
      'dtype': 'float32',
      'param_dtype': 'float32',
      'output_tokens': True,
      'use_dense_general': False,
      'pool_type': 'gap'
  })
  config.model.text = ConfigDict({
      'variant': arg.txt_name,
      'pool_type': 'last',
      'use_flash_attn': False,
      'remat_policy':arg.remat,
      'casual_mask': False,
      'scan_mlp': False,
      'scan_attn': False,
      'mlp_chunck': 128,
      'head_zeroinit': False,
      'dtype': 'float32',
      'param_dtype': 'float32',
      'vocab_size': arg.vocab_size,
      "embed_cls": True,
      "output_tokens": True,
      'use_dense_general': False
  })
  config.model.temperature_init = 1/0.07
  
  # hack for sovit
  if arg.use_sovit:
      dim = 1152
  else:
      dim = {'T': 192, 'S':384, 'B': 512, 'L': 768, 'H':1024, 'g':1024}[arg.img[0]]

 # dim = 768
  config.model.out_dim = (dim if arg.img_head else None, dim)  # (image_out_dim, text_out_dim)

  config.optax_name = 'scale_by_adam'
  config.ft_from = ""
  config.load_transform=""
  config.masked_init = ""


  config.input.batch_size = int(1024 * 16 * arg.batch_factor)
  batch_size = int(1024 * 16 * arg.batch_factor)

  imagenet_samples = 1281167
  vitual_imagenet_epoch = arg.imagenet_epoch
  vitual_warmup_epoch = arg.vitual_warmup_epoch
  total_seen_samples = imagenet_samples * vitual_imagenet_epoch
  total_warmup_samples = imagenet_samples* vitual_warmup_epoch

  config.total_steps = int(total_seen_samples // batch_size) if not arg.runlocal else 1
  config.lr = arg.base_lr * 64 * arg.batch_factor # lr for 256
  config.wd = 0.2
  warmup_steps = int(total_warmup_samples // batch_size) # for 16k batch size  # max(int(0.03 * config.total_epochs), 100)
  config.schedule = [
      #('img/.*', None),  # Freezes image tower.
      ('.*', dict(decay_type='cosine', warmup_steps=warmup_steps, min_lr=0, max_lr=arg.base_lr * 64 * arg.batch_factor)),
  ]

  config.optax = dict(mu_dtype='bfloat16',  b1=0.9,  b2=0.95)


  config.loss_type = 'coca'
  config.coca_caption_loss_weight = 2
  config.clip_loss_weight = 1
  config.loss_use_global_batch = True
  config.local_loss = True
  config.mask_ratio = 0.
  config.cpu_unit8 = True



  # Eval section (Both few-shot and zero-shot)
  config.eval_only = False
  eval_common = dict(
      type='proj.image_text.contrastive',
      use_global_batch=config.loss_use_global_batch,
      log_steps=int(2000 // arg.batch_factor),
  )

  config.evals = {}

  sub = '[:4]' if arg.runlocal else ''

  tokenizer_eval = lambda inkey: (
    f'my_eval_bert_tokenize(inkey="{inkey}", max_len={arg.token_len}, '
    f'vocab_path="{vocab_path}", add_bos=True, add_eos=True)'
    )
  
  config.evals.disclf = {}
  config.evals.disclf.dataset_names = ['imagenet2012']
  config.evals.disclf.split = f'validation{sub}'
  config.evals.disclf.data_dir = 'your data dir'
  config.evals.disclf.pp_img = f'|resize_small({arg.res}, method="bilinear", antialias=True)|central_crop({arg.res})|vgg_value_range'
  config.evals.disclf.pp_txt = tokenizer_eval('texts')
  config.evals.disclf.canonicalize = True
  config.evals.disclf.first_class_name_only = False
  config.evals.disclf.type = 'proj.image_text.discriminative_classifier'
  config.evals.disclf.prefix = 'z/0shot/'
  config.evals.disclf.log_steps = eval_common['log_steps']

  # retrieval
  config.evals.retrieval = {}
  config.evals.retrieval = dict(log_steps=eval_common['log_steps'], type='proj.image_text.retrieval')
  config.evals.retrieval.dataset = 'coco_captions'
  config.evals.retrieval.split = 'val'
  config.evals.retrieval.data_dir = 'your data dir'
  config.evals.retrieval.txt_name = ('captions', 'text')
  # Note that initial "decode|" is not needed.
  config.evals.retrieval.pp_img = f'|resize_small({arg.res}, method="bilinear", antialias=True)|central_crop({arg.res})|vgg_value_range'
  config.evals.retrieval.pp_txt = tokenizer_eval('texts')

  # flickr30k retrieval
  config.evals.retrieval_flikr = {}
  config.evals.retrieval_flikr = dict(log_steps=eval_common['log_steps'], type='proj.image_text.retrieval')
  config.evals.retrieval_flikr.dataset = 'flickr30k'
  config.evals.retrieval_flikr.split = 'test'
  config.evals.retrieval_flikr.data_dir = 'your data dir'
  config.evals.retrieval_flikr.txt_name = 'captions'
  # Note that initial "decode|" is not needed.
  config.evals.retrieval_flikr.pp_img = f'|resize_small({arg.res}, method="bilinear", antialias=True)|central_crop({arg.res})|vgg_value_range'
  config.evals.retrieval_flikr.pp_txt = tokenizer_eval('texts')

  config.seed = 0
  config.l = config.m = 0

  config.hf_upload = dict(
      repo_name='custmozie a model name of the repo',
      save_directory='your local vm saving path',
      token='hf token for uploading',
      commit_message='first commit',
      model_id='UCSC-VLAA/ViT-L-14-CLIPS-224-Recap-DataComp-1B',  # templete_model config/tokenizer
      cache_dir='/home/lixianhang/hf_cache/',
  )

  return config
