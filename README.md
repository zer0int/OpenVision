# OpenVision: Advanced Vision-Language Models Training

This repository contains the code for training and fine-tuning vision-language models based on the OpenVision framework. It provides a scalable and efficient approach to training multimodal models on TPU infrastructure.

## Features

- Optimized for Google Cloud TPU training
- Supports various encoder architectures (ViT models of different sizes)
- Implements efficient training strategies including model sharding
- Supports pre-training and multi-stage fine-tuning 
- Compatible with CLIP-style vision-language training

## Installation

### Requirements

```bash
# Clone the repository
git clone https://github.com/UCSC-VLAA/OpenVision.git
cd openvision

bash setup.sh  DEVICE=tpu JAX_VERSION=0.4.38
```

### TPU VM Setup

This codebase is optimized for TPU VM instances. To set up a TPU VM:

1. Create a TPU VM instance on Google Cloud
2. Install JAX with TPU support
3. Clone the repository and install dependencies

## Training

The training process is divided into two main phases:

1. **Pre-training**: Training the model on large-scale datasets with smaller resolution
2. **Fine-tuning**: Refining the model on higher resolution images

## TPU Pod Training

### Setting Up and Using the tpu_command.sh Tool

`tpu_command.sh` is a convenient tool for synchronizing files between different VMs and managing TPU resources.

#### Prerequisites
- Add your SSH key to your Google Cloud project's metadata, and GCP will automatically propagate these SSH keys to all VMs

#### Basic Usage
1. In your local terminal, run: `. ./tpu_command.sh`
2. Enter `tpu` to launch the interactive menu
3. Select the function you need:
   - **ssh**: Connect to TPU VM
   - **sync dir**: Synchronize directories to all VMs in the TPU Pod
   - **kill job**: Terminate running jobs
   - **prepare env**: Set up the TPU environment
   - **check**: Check if TPU core is occupied
   - **rm tpu logs**: Clear TPU logs
   - **exit**: Exit the tool

#### Workflow Example
1. Select the `sync dir` function, then choose your TPU region, and follow the prompts to upload your code to the Pod
2. Use `prepare_env` to set up the environment and synchronize all files
3. Connect to the VM using `ssh`
4. Start a tmux session on the VM
5. Execute the `train.sh` script in the tmux session

### Training Parameters

The main parameters for training can be adjusted in the `scripts/project/openvision/train.sh` script:

- `TPU_NAME`: Name of your TPU VM instance
- `ZONE`: Google Cloud zone where your TPU is located
- `PROJECT_ID`: Your Google Cloud project ID
- `EXP_PATH`: Google Cloud Storage path for experiment outputs
- `BATCH_FACTOR`: Controls the effective batch size
- `PRE_TRAIN_EPOCH`/`FT_TRAIN_EPOCH`: Number of epochs for pre-training and fine-tuning
- `PRE_LR`/`FT_LR`: Learning rates for pre-training and fine-tuning
- `PRE_RES`/`FT_RES`: Image resolutions for pre-training and fine-tuning
- `MODEL`/`TXT_MODEL`: Architecture variants for image and text encoders

### Training Commands

To start training with the default settings:

```bash
# Run pre-training
bash scripts/project/openvision/train.sh
```

The script automatically handles:
1. Pre-training with low resolution (default: 84px/160px)
2. Fine-tuning with medium resolution (default: 224px)
3. Optional second fine-tuning with high resolution (default: 384px/336px)

## Data Preparation

Prepare your data according to the format expected by the input pipeline. The training script expects the following datasets:

1. Main training data (e.g., DataComp dataset)
2. Evaluation datasets:
   - ImageNet for classification
   - COCO for image-text retrieval
   - Flickr30k for additional retrieval evaluation

### Preparing Data Paths

Update the following variables in the training script to point to your datasets:

```bash
export IN1K_DATA_DIR=gs://your-bucket/imagenet
export COCO_DATA_DIR=gs://your-bucket/coco
export Flickr_DATA_DIR=gs://your-bucket/flickr30k
export DATACOMP_PATH=gs://your-bucket/datacomp/shards
```

## Customization

### Model Architecture

The framework supports different model architectures which can be specified in the training script:

- Vision models: 'Ti/16', 'S/16', 'B/16', 'L/16' (for Tiny, Small, Base, Large variants)
- Text models: Same variants are available for text encoders
- Text decoder (if needed): Controlled by the `DECODER_NAME` parameter

### Training Configuration

Training configurations are defined in `src/configs/openvision.py`. You can:

1. Modify sharding strategies for distributed training
2. Change optimization parameters
3. Adjust data augmentation and tokenization settings
4. Configure evaluation metrics and checkpointing

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details. 