#!/bin/bash

# Copyright 2019 Mobvoi Inc. All Rights Reserved.
. ./path.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="1"
# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=0 # start from 0 if you need to start from data preparation
stop_stage=5

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0
# The aishell dataset location, please change this to your own path
# make sure of using absolute path. DO-NOT-USE relatvie path!
# data=/export/data/asr-data/OpenSLR/33/
data=/hdd/dataset/commonvoice_zh-tw/cv-corpus-11.0-2022-09-21/zh-TW
data_url=www.openslr.org/resources/33

nj=16
dict=data/dict/lang_char.txt

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train
dev_set=dev
test_set=test
# Optional train_config
# 1. conf/train_transformer.yaml: Standard transformer
# 2. conf/train_conformer.yaml: Standard conformer
# 3. conf/train_unified_conformer.yaml: Unified dynamic chunk causal conformer
# 4. conf/train_unified_transformer.yaml: Unified dynamic chunk transformer
# 5. conf/train_u2++_conformer.yaml: U2++ conformer
# 6. conf/train_u2++_transformer.yaml: U2++ transformer
checkpoint_dir=20220506_u2pp_conformer_exp
train_config=${checkpoint_dir}/train.yaml
cmvn=true
dir=exp/conformer
checkpoint=${checkpoint_dir}/final.pt
checkpoint_dict=${checkpoint_dir}/units.txt

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10
decode_modes="attention_rescoring ctc_greedy_search"

. tools/parse_options.sh || exit 1;

set -u
set -o pipefail


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Data preparation"
  python3 local/commonvoice_data_prep.py \
    --data_dir ${data} \
    --output_dir data/ \
    --split train dev test
fi

dict=data/dict/lang_char.txt

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    tools/compute_cmvn_stats.py --num_workers 16 --train_config $train_config \
    --in_scp data/${train_set}/wav.scp \
    --out_cmvn data/$train_set/global_cmvn
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Make a dictionary"
  mkdir -p $(dirname $dict)
  echo "<blank> 0" > ${dict}  # 0 is for "blank" in CTC
  echo "<unk> 1"  >> ${dict}  # <unk> must be 1
  tools/text2token.py -s 1 -n 1 data/train/text | cut -f 2- -d" " \
    | tr " " "\n" | sort | uniq | grep -a -v -e '^\s*$' | \
    awk '{print $0 " " NR+1}' >> ${dict}
  num_token=$(cat $dict | wc -l)
  echo "<sos/eos> $num_token" >> $dict
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Prepare data, prepare required format"
  for x in dev test ${train_set}; do
    if [ $data_type == "shard" ]; then
      tools/make_shard_list.py --num_utts_per_shard $num_utts_per_shard \
        --num_threads 16 data/$x/wav.scp data/$x/text \
        $(realpath data/$x/shards) data/$x/data.list
    else
      tools/make_raw_list.py data/$x/wav.scp data/$x/text \
        data/$x/data.list
    fi
  done
fi



if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Start training"
  mkdir -p $dir
  # INIT_FILE is for DDP synchronization
  INIT_FILE=$dir/ddp_init
  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  dist_backend="nccl"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp data/${train_set}/global_cmvn $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"
  # train.py will write $train_config to $dir/train.yaml with model input
  # and output dimension, train.yaml will be used for inference or model
  # export later
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    python wenet/bin/train.py --gpu $gpu_id \
      --config $train_config \
      --data_type "raw" \
      --symbol_table $checkpoint_dict \
      --train_data data/$train_set/data.list \
      --cv_data data/dev/data.list \
      ${checkpoint:+--checkpoint $checkpoint} \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      $cmvn_opts \
      --num_workers 8 \
      --pin_memory
  } &
  done
  wait
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Test model"
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    python wenet/bin/average_model.py \
        --dst_model $decode_checkpoint \
        --src_path $dir  \
        --num ${average_num} \
        --val_best
  fi
  # Specify decoding_chunk_size if it's a unified dynamic chunk trained model
  # -1 for full chunk
  decoding_chunk_size=
  ctc_weight=0.5
  reverse_weight=0.0
  for testset in ${test_set} ${dev_set}; do
  {
    for mode in ${decode_modes}; do
    {
      base=$(basename $decode_checkpoint)
      result_dir=$dir/${testset}_${mode}_${base}
      mkdir -p $result_dir
      python wenet/bin/recognize.py --gpu 0 \
        --mode $mode \
        --config $dir/train.yaml \
        --data_type "raw" \
        --test_data data/$testset/data.list \
        --checkpoint $decode_checkpoint \
        --beam_size 10 \
        --batch_size 1 \
        --penalty 0.0 \
        --dict $checkpoint_dict \
        --ctc_weight $ctc_weight \
        --reverse_weight $reverse_weight \
        --result_file $result_dir/text \
        ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
      python tools/compute-wer.py --char=1 --v=1 \
        data/$testset/text $result_dir/text > $result_dir/wer
    }
    done
    wait
  }
  done
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Export the best model you want"
  python wenet/bin/export_jit.py \
    --config $dir/train.yaml \
    --checkpoint $dir/avg_${average_num}.pt \
    --output_file $dir/final.zip
fi
