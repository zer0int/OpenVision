#!/bin/bash

# to_convert:
FT_MODEL_INIT=gs://lxh_jaxtpu_eu_ckpt/llava_xl_clip/datacomp1b/tpu-v3-256-pod-vm-spot/clips_Base16_pretrain_10000_160_bs_32k_run_0/B/16_B/16_1.0_clips_Base16_pretrain_10000_160_bs_32k_run_0_bz_2_fp32_pre_10000_concat_False_prob_1.0/ft/B/16_B/16_1.0_clips_Base16_pretrain_10000_160_bs_32k_run_0_fp32_ft_400_concat_False_prob_1.0
FT_MODEL_INIT=gs://lxh_jaxtpu_eu_ckpt/llava_xl_clip/datacomp1b/tpu-v3-256-pod-vm-spot/clips_Base16_pretrain_10000_160_bs_32k_run_0/B/16_B/16_1.0_clips_Base16_pretrain_10000_160_bs_32k_run_0_bz_2_fp32_pre_10000_concat_False_prob_1.0/ft/B/16_B/16_1.0_clips_Base16_pretrain_10000_160_bs_32k_run_0_fp32_ft_800_concat_False_prob_1.0
FT_MODEL_INIT=gs://lxh_jaxtpu_eu_ckpt/llava_xl_clip/datacomp1b/tpu-v3-256-pod-vm-spot/clips_Base16_336_200e_from_224_800e/B/16_B/16_1.0_clips_Base16_336_200e_from_224_800e_bz_2_fp32_pre_10000_concat_False_prob_1.0/ft/B/16_B/16_1.0_clips_Base16_336_200e_from_224_800e_fp32_ft_200_concat_False_prob_1.0
FT_MODEL_INIT=gs://lxh_jaxtpu_eu_ckpt/llava_xl_clip/datacomp1b/tpu-v3-256-pod-vm/LOAD_BASE_DECODER_para_32_clips_Base16_384_200e_pretrain_10000_160_bs_32k_run_1/B/16_B/16_1.0_LOAD_BASE_DECODER_para_32_clips_Base16_384_200e_pretrain_10000_160_bs_32k_run_1_bz_2_fp32_pre_10000_concat_False_prob_1.0/ft/B/16_B/16_1.0_LOAD_BASE_DECODER_para_32_clips_Base16_384_200e_pretrain_10000_160_bs_32k_run_1_fp32_ft_200_concat_False_prob_1.0
FT_MODEL_INIT=gs://lxh_jaxtpu_eu_ckpt/llava_xl_clip/datacomp1b/tpu-v3-128-pod-vm-spot/LOADED_DECODER_FT_384_8K_clips_Ti16_pretrain_10000_160_bs_32k_run_0/Ti/16_Ti/16_1.0_LOADED_DECODER_FT_384_8K_clips_Ti16_pretrain_10000_160_bs_32k_run_0_bz_2_fp32_pre_10000_concat_False_prob_1.0/ft/Ti/16_Ti/16_1.0_LOADED_DECODER_FT_384_8K_clips_Ti16_pretrain_10000_160_bs_32k_run_0_fp32_ft_200_concat_False_prob_1.0

token_len=128


# vit text encoder
export JAX_PLATFORMS='cpu'

# vit text encoder
img=Ti/16
txt_name=Ti/16
txt_decoder_name=Ti
res=384
python3 -m src.convert_upload.transfer_jax2hf \
--config=src/configs/openvision.py:\
res=${res},img=${img},txt_name=${txt_name},txt_decoder_name=${txt_decoder_name},token_len=${token_len},use_openclip_tokenizer=False,\
data_parallelism=1,fsdp_parallelism=1,tensor_parallelism=1 \
--workdir=${FT_MODEL_INIT} \
--config.model.text.pool_type='last' \
--config.model.image.dtype=float32 \
--config.model.image.param_dtype=float32 \
--config.model.text.dtype=float32 \
--config.model.text.param_dtype=float32 \
--config.hf_upload.repo_name="UCSC-VLAA/openvision-vit-tiny-patch16-384" \
--config.hf_upload.save_directory="/home/lixianhang/jax2hf/openvision-vit-tiny-patch16-384" \
--config.hf_upload.token="YOUR_HF_TOKEN" \
--config.hf_upload.commit_message="firstcommit" \
--config.hf_upload.cache_dir="/home/lixianhang/hf_cache/" \
--config.model.text.use_dense_general=False \
--config.model.image.use_dense_general=False \









