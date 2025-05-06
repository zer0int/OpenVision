#!/bin/bash


export ZONE="your-zone"                          # Region
export PROJECT_ID="your-project-id"                   # Project ID
export TPU_NAME="your-tpu-name"                   # TPU Instance Name
export ACCOUNT="your-service-account"  # Service Account

# Print configuration information
echo "Project ID: $PROJECT_ID"
echo "Zone: $ZONE"
echo "TPU Name: $TPU_NAME"

# Create TPU instance
# If creation fails, it will automatically retry
while true; do
    gcloud alpha compute tpus tpu-vm create $TPU_NAME \
        --project $PROJECT_ID \
        --zone=$ZONE \
        --spot \                                # Use spot instance (cheaper but may be preempted)
        --accelerator-type=your-accelerator-type \            # TPU type and size
        --version=your-tpu-software-version \             # TPU software version
        --service-account=$ACCOUNT
    
    echo "TPU creation command executed, retrying in 5 seconds..."
    sleep 5                                    # Retry interval
done 