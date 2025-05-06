export  ZONE="your-zone"
export PROJECT_ID="your-project-id"
export TPU_NAME="your-tpu-name"


set -e

############################################################
# 
#############################################################
export IN1K_DATA_DIR="your-in1k-data-dir"
export COCO_DATA_DIR="your-coco-data-dir"
export DATACOMP_PATH="your-datacomp-path"
export SPLIT='full'
 export EXP_PATH="your-exp-path"
export Flickr_DATA_DIR="your-flickr-data-dir"
export ABLATION_NAME="your-ablation-name"
export BATCH_FACTOR=2  # pre-training batch size is 32k=BATCH_FACTOR*32
export FT_BATCH_FACTOR=1  # fine-tuning batch size is 32k=FT_BATCH_FACTOR*32
export PRE_TRAIN_EPOCH=10000  # small image pre-training In1k epoch =10000*1.28M=12.8M
export PRE_WARMUP_EPOCH=40  # pre-training warmup epoch
export FT_TRAIN_EPOCH=800  # large image fine-tuning epoch = 800*1.28M=1.024M
export FT_WARMUP_EPOCH=20  # fine-tuning warmup epoch   
export data_parallelism=128  # data parallelism adjust to your TPU core number
export fsdp_parallelism=1
export tensor_parallelism=1
export PRE_LR=8e-6
export FT_LR=4e-7
export PRE_RES=84
export FT_RES=224
export CONCAT_CAP=False
export MIX_PROB=1.0
export MODEL='S/16' # can be S/16, B/16, L/16, L/14, SoVit400m/14, H/14
export TXT_MODEL='S/16' # can be S/16, B/16, L/16, L/14,SoVit400m/14, H/14
export DECODER_NAME='S' # can be S, B, L, SoVit, H
export WANDB_PROJ="your-wandb-project"
export WANDB_ENTITY="your-wandb-entity"
export mask_ratio=0.
export token_len=80 # text token length
export output_token_len=128 # text decoutput token length
export PRE_WANDB_NAME=${MODEL}_${TXT_MODEL}_${MIX_PROB}_${ABLATION_NAME}_bz_${BATCH_FACTOR}_fp32_pre_${PRE_TRAIN_EPOCH}_concat_${CONCAT_CAP}_prob_${MIX_PROB}
export FT_WANDB_NAME=${MODEL}_${TXT_MODEL}_${MIX_PROB}_${ABLATION_NAME}_fp32_ft_${FT_TRAIN_EPOCH}_concat_${CONCAT_CAP}_prob_${MIX_PROB}
export PRE_WORK_DIR=${EXP_PATH}/${ABLATION_NAME}
export FT_WORK_DIR=${PRE_WORK_DIR}/ft_224
export FT_MODEL_INIT=${PRE_WORK_DIR}/checkpoint.npz
export TRAIN_CONFIG=src/configs/openvision.py
export VOCAB_PATH=assets/bert_base_vocab_bos_eos.txt


###pre-training script
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd openvision && \
. ~/openvision/openvision_env/bin/activate && wandb login 4348a91d800a8e8eb33b86c30197241bb228e268 && \
python3 -m src.main_clip \
--config=${TRAIN_CONFIG}:\
token_len=${token_len},\
output_token_len=${output_token_len},\
res=${PRE_RES},img=${MODEL},mask_ratio=${mask_ratio},txt_name=${TXT_MODEL},\
txt_decoder_name=${DECODER_NAME},\
base_lr=${PRE_LR},vocab_path=${VOCAB_PATH},\
concat=${CONCAT_CAP},prob=${MIX_PROB},\
batch_factor=${BATCH_FACTOR},\
data_parallelism=${data_parallelism},fsdp_parallelism=${fsdp_parallelism},tensor_parallelism=${tensor_parallelism},\
imagenet_epoch=${PRE_TRAIN_EPOCH},\
vitual_warmup_epoch=${PRE_WARMUP_EPOCH} \
--workdir=${PRE_WORK_DIR} \
--config.eval_only=False \
--config.wandb.log_wandb=True \
--config.model.image.use_flash_attn=False \
--config.model.text.use_flash_attn=False \
--config.model.image.dtype=bfloat16 \
--config.model.text.dtype=bfloat16 \
--config.model.text_decoder_config.dtype=bfloat16 \
--config.model.text.param_dtype=float32 \
--config.wandb.experiment=${PRE_WANDB_NAME} \
--config.wandb.project=${WANDB_PROJ} \
--config.wandb.entity=${WANDB_ENTITY} \
--config.input.data.data_dir=${DATACOMP_PATH} \
--config.input.data.split=${SPLIT} \
--config.evals.disclf.data_dir=${IN1K_DATA_DIR} \
--config.evals.retrieval.data_dir=${COCO_DATA_DIR} \
--config.evals.retrieval_flikr.data_dir=${Flickr_DATA_DIR} \
"



gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd openvision && \
. ~/openvision/openvision_env/bin/activate && wandb login [your-wandb-token] && \
python3 -m src.main_clip \
--config=${TRAIN_CONFIG}:\
res=${FT_RES},mask_ratio=${mask_ratio},img=${MODEL},vocab_path=${VOCAB_PATH},txt_name=${TXT_MODEL},base_lr=${FT_LR},\
data_parallelism=${data_parallelism},fsdp_parallelism=${fsdp_parallelism},tensor_parallelism=${tensor_parallelism},\
concat=${CONCAT_CAP},prob=${MIX_PROB},\
batch_factor=${FT_BATCH_FACTOR},\
txt_decoder_name=${DECODER_NAME},\
imagenet_epoch=${FT_TRAIN_EPOCH},\
vitual_warmup_epoch=${FT_WARMUP_EPOCH} \
--workdir=${FT_WORK_DIR} \
--config.wandb.log_wandb=True \
--config.ft_from=${FT_MODEL_INIT} \
--config.wandb.experiment=${FT_WANDB_NAME} \
--config.model.image.dtype=bfloat16 \
--config.model.text.dtype=bfloat16 \
--config.model.text_decoder_config.dtype=bfloat16 \
--config.model.image.param_dtype=float32 \
--config.model.text.param_dtype=float32 \
--config.wandb.project=${WANDB_PROJ} \
--config.wandb.entity=${WANDB_ENTITY} \
--config.input.data.data_dir=${DATACOMP_PATH} \
--config.input.data.split=${SPLIT} \
--config.evals.disclf.data_dir=${IN1K_DATA_DIR} \
--config.evals.retrieval.data_dir=${COCO_DATA_DIR} \
--config.evals.retrieval_flikr.data_dir=${Flickr_DATA_DIR} \
"



# 384
export IN1K_DATA_DIR="your-in1k-data-dir"
export COCO_DATA_DIR="your-coco-data-dir"
export Flickr_DATA_DIR="your-flickr-data-dir"
export DATACOMP_PATH="your-datacomp-path"
export SPLIT='full'
export EXP_PATH="your-exp-path"
export ABLATION_NAME="your-ablation-name"
export BATCH_FACTOR=2
export FT_BATCH_FACTOR=0.5 # fine-tuning batch size is 8k=16k*0.5
export PRE_TRAIN_EPOCH=10000
export PRE_WARMUP_EPOCH=40
export FT_TRAIN_EPOCH=200 # 384 or 336 large image fine-tuning epoch = 200*1.28M=256M
export FT_WARMUP_EPOCH=0
export data_parallelism=128
export fsdp_parallelism=1
export tensor_parallelism=1
export PRE_LR=8e-6
export FT_LR=1e-7
export PRE_RES=84
export FT_RES=384 # change resolution to 336 for ViT-L/14 and ViT-H/14
export CONCAT_CAP=False
export MIX_PROB=1.0
export MODEL='S/16'
export TXT_MODEL='S/16'
export DECODER_NAME='S'
export WANDB_PROJ="your-wandb-project"
export WANDB_ENTITY="your-wandb-entity"
export mask_ratio=0.
export token_len=80
export output_token_len=128
export PRE_WANDB_NAME=${MODEL}_${TXT_MODEL}_${MIX_PROB}_${ABLATION_NAME}_bz_${BATCH_FACTOR}_fp32_pre_${PRE_TRAIN_EPOCH}_concat_${CONCAT_CAP}_prob_${MIX_PROB}
export FT_WANDB_NAME=${MODEL}_${TXT_MODEL}_${MIX_PROB}_${ABLATION_NAME}_fp32_ft_${FT_TRAIN_EPOCH}_concat_${CONCAT_CAP}_prob_${MIX_PROB}
export PRE_WORK_DIR=${EXP_PATH}/${ABLATION_NAME}
export FT_WORK_DIR=${PRE_WORK_DIR}/ft_384
export FT_MODEL_INIT=${PRE_WORK_DIR}/ft_224/checkpoint.npz
export TRAIN_CONFIG=src/configs/openvision.py
export VOCAB_PATH=assets/bert_base_vocab_bos_eos.txt


##FT-training script
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd openvision && \
. ~/openvision/openvision_env/bin/activate && wandb login [your-wandb-token] && \
python3 -m src.main_clip \
--config=${TRAIN_CONFIG}:\
res=${FT_RES},mask_ratio=${mask_ratio},img=${MODEL},vocab_path=${VOCAB_PATH},txt_name=${TXT_MODEL},base_lr=${FT_LR},\
data_parallelism=${data_parallelism},fsdp_parallelism=${fsdp_parallelism},tensor_parallelism=${tensor_parallelism},\
concat=${CONCAT_CAP},prob=${MIX_PROB},\
batch_factor=${FT_BATCH_FACTOR},\
txt_decoder_name=${DECODER_NAME},\
imagenet_epoch=${FT_TRAIN_EPOCH},\
vitual_warmup_epoch=${FT_WARMUP_EPOCH} \
--workdir=${FT_WORK_DIR} \
--config.wandb.log_wandb=True \
--config.ft_from=${FT_MODEL_INIT} \
--config.wandb.experiment=${FT_WANDB_NAME} \
--config.model.image.dtype=bfloat16 \
--config.model.text.dtype=bfloat16 \
--config.model.text_decoder_config.dtype=bfloat16 \
--config.model.image.param_dtype=float32 \
--config.model.text.param_dtype=float32 \
--config.wandb.project=${WANDB_PROJ} \
--config.wandb.entity=${WANDB_ENTITY} \
--config.input.data.data_dir=${DATACOMP_PATH} \
--config.input.data.split=${SPLIT} \
--config.evals.disclf.data_dir=${IN1K_DATA_DIR} \
--config.evals.retrieval.data_dir=${COCO_DATA_DIR} \
--config.evals.retrieval_flikr.data_dir=${Flickr_DATA_DIR} \
"

