#!/bin/bash

# to_convert:
FT_MODEL_INIT="your_ckpt_gcs_path"


# vit text encoder
export JAX_PLATFORMS='cpu'

# vit text encoder
img=Ti/16
txt_name=Ti/16
txt_decoder_name=Ti
res=384
token_len=128



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
--config.hf_upload.repo_name="YOUR_REPO_NAME" \
--config.hf_upload.save_directory="YOUR_SAVE_DIRECTORY" \
--config.hf_upload.token="YOUR_HF_TOKEN" \
--config.hf_upload.commit_message="YOUR_COMMIT_MESSAGE" \
--config.hf_upload.cache_dir="YOUR_CACHE_DIRECTORY" \
--config.model.text.use_dense_general=False \
--config.model.image.use_dense_general=False \









