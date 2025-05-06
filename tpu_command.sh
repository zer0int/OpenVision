#!/bin/bash


# !!!!you can first add the ssh key file in the project metadat ssh-keys
ALL_PROJECTS=("focus-album-323718" "priors-medical-ai")
ALL_ZONE=("us-central2-b"  "europe-west4-a" "us-east1-d")


export USER="your_username"
export SSH_KEY_DIR=$HOME/.ssh/keys # the directory of your ssh keys
export CODEBASE=openvision # the codebase name
export PROJECT_DIR=/Users/your_username/PycharmProjects/$CODEBASE # the project directory on your local machine
export TARGET_DIR=/home/$USER/ # the target directory on the TPU


function echo_sleep {
     echo $1
     sleep 2
}

function choose_zone {
  echo "choose the project"
  select proj in ${ALL_PROJECTS[@]}; do
      echo "you have choose project $proj"
      export PROJECT_ID=$proj
      break
  done
  echo "choose the zone"
  select zone in ${ALL_ZONE[@]}; do
      echo "you have choose zone $zone"
      export ZONE=$zone
      break
  done


}

function prepare_env {
  #select vm
  choose_vm
  echo_sleep "prepar env on vm $TPU_NAME"
  gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
 --command "cd  $CODEBASE && bash setup.sh  DEVICE=tpu JAX_VERSION=0.4.38"
}


function rm_tpu_logs {
  #select vm
  choose_vm

  gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
  --command "sudo rm -rf /tmp/tpu_logs/ "
   gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
   --command "sudo rm  -rf /tmp/libtpu_lockfile/ "

   gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
  --command "sudo rm -rf /var/crash/ "

  gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo rm -rf /tmp/libtpu_lockfile"

  gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo chmod -R 777  /tmp/tpu_logs"
}


function ssh_vm {
    choose_vm
    gcloud alpha compute tpus tpu-vm ssh  $USER@$TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=0
}


function sync_dirs {

    choose_vm
    # we need you private key for sync
    PRIVATE_KEY=$1
    PROJECT_DIR=$2
    VM_DIR=$3
    #gcloud alpha compute tpus tpu-vm scp --recurse  $PROJECT_DIR $TPU_NAME:$VM_DIR --zone=$ZONE --worker=all --project $PROJECT_ID
    while IFS= read -r line; do
        TPU_IPS+=("$line")
    done < <(gcloud alpha compute tpus tpu-vm describe $TPU_NAME --zone $ZONE --project $PROJECT_ID | grep -e "externalIp"|grep -oE "[0-9]{1,3}(\.[0-9]{1,3}){3}")

    echo_sleep $USER@${TPU_IPS[@]}

    for tpu_ip in ${TPU_IPS[@]}; do
        rsync -avPI -e "ssh -i $PRIVATE_KEY -o StrictHostKeyChecking=no" \
              --exclude=logs --exclude=__pycache__ \
              --exclude=.git \
              $PROJECT_DIR $USER@$tpu_ip:$VM_DIR ;
    done
    unset TPU_IPS
}

function choose_vm {
    choose_zone
    VM_LIST=()
    # read the tpu vm list name
    while IFS= read -r line; do
        VM_LIST+=( "$line" )
    done < <(gcloud alpha compute tpus tpu-vm list --zone $ZONE --project $PROJECT_ID --format="value(name)")

    echo Now select ${VM_LIST[@]} you are requested
    # select one of vm to copy ssh_keys, sync file or directly ssh connect
    select VM in ${VM_LIST[@]}; do
        export TPU_NAME=$VM
        echo_sleep "you have selected $VM"
        break
    done
    unset  VM_LIST
}

function run_jobs {

    #select vm
    choose_vm

    CONFIG=$1
    COMMAND="cd $CODEBASE && bash scripts/run.sh $CONFIG"
    echo $COMMAND
    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command "tmux new -d -s launch"

    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command "tmux send-keys -t launch '$COMMAND' C-m"

}

function kill_jobs {

    #select vm
    choose_vm

    COMMAND_1="sudo pkill -f  python3"
    COMMAND_2="sudo pkill -f python"
    COMMAND_5="pkill -f  python3"
    COMMAND_6="pkill -f python"
    COMMAND_3="sudo lsof -w /dev/accel0"
    COMMAND_4="sudo lsof -t /dev/accel0 | xargs -r -I {} sudo kill -9 {} "


    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command $COMMAND_4

    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command $COMMAND_5

    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command  $COMMAND_6

     gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command  $COMMAND_2

    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command $COMMAND_1

    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command $COMMAND_2

    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command $COMMAND_1

      gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command $COMMAND_3

}

function check {

    #select vm
    choose_vm

    COMMAND_3="sudo lsof -w /dev/accel0"

    echo COMMAND_3


    gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
    --command $COMMAND_3

}

function tpu {

    echo "select what you want to do"

    # select function that you will execute
    select func in  ssh_vm sync_dirs kill_jobs prepare_env check  rm_tpu_logs exit; do
          while IFS= read -r line; do
                  pub_keys+=("$line")
          done < <(ls $SSH_KEY_DIR)

          if [ $func = "sync_dirs" ] || [ $func = "prepare_env" ];
          then
             echo "remember to copy keys first"
             echo "now sync $PROJECT_DIR to $TARGET_DIR"
             echo "select the private key to sync file: "
             select private in $pub_keys; do
                $func $SSH_KEY_DIR/$private $PROJECT_DIR $TARGET_DIR
             break
             done
             unset pub_keys
          elif [ $func = "exit" ] ;
          then
              unset pub_keys
              break
          else
             # ssh selected vm
             unset pub_keys
             echo "now we $func"
             $func
          fi
    break
    done
}

unset  VM_LIST
unset pub_keys
export -f sync_dirs ssh_vm tpu  prepare_env run_jobs kill_jobs check rm_tpu_logs