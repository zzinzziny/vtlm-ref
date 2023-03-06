#!/bin/bash

# Takes a VTLM checkpoint and finetunes it for Multi30k MMT (with visual features)

DATA_PATH="/data/IMT/VTLM/data/multi30k"
FEAT_PATH="/data/IMT/VTLM/data/multi30k/features"
DUMP_PATH="/data/IMT/VTLM/models"

CKPT="/data/IMT/VTLM/models/vtlm-on-multi30k/zt6du0g3zo/best-valid_en_de_mlm_ppl.pth"

if [ -z $CKPT ]; then
  echo 'You need to provide a checkpoint .pth file for pretraining'
  exit 1
fi

shift 1

# sth like periodic-xxx.pth or best-...pth
CKPT_NAME=`basename $CKPT | tr -- '-.' '_'`
CKPT_ID=$(basename `dirname $CKPT`)
CKPT_NAME="${CKPT_ID}_${CKPT_NAME}"

LOG="`dirname $CKPT`/zt6du0g3zo/train.log"
# Fetch previous args
PREV_ARGS=`egrep '(emb_dim|n_layers|n_heads):' $LOG | sed 's#\s*\([a-z_]*\): \([0-9]*\)$#--\1 \2#'`

PAIR=$(basename `ls ${DATA_PATH}/train.*pth | head -n1` | cut -d'.' -f2)
L1=`echo $PAIR | cut -d'-' -f1`
EPOCH=`wc -l ${DATA_PATH}/train.${PAIR}.$L1 | head -n1 | cut -d' ' -f1`
BS=${BS:-64}
LR=${LR:-0.00001}
NAME="${CKPT_NAME}_ftune_mmt_vtlm_bs${BS}_lr${LR}"
PREFIX=${PREFIX:-}
DUMP_PATH="${DUMP_PATH}/${PREFIX}"

export CUDA_VISIBLE_DEVICES=0
export NGPU=1

python -m torch.distributed.launch --nproc_per_node=$NGPU train.py \
  --beam_size 8 --exp_name ${NAME} --dump_path ${DUMP_PATH} \
  --reload_model ${CKPT},${CKPT} --data_path ${DATA_PATH} --encoder_only false \
  --lgs 'en-de' --mmt_step "en-de" $PREV_ARGS \
  --dropout '0.2' --attention_dropout '0.1' --gelu_activation true \
  --batch_size ${BS} --optimizer "adam,lr=${LR}" \
  --epoch_size ${EPOCH} --eval_bleu true --max_epoch 500 \
  --stopping_criterion 'valid_en-de_mmt_bleu,20' --validation_metrics 'valid_en-de_mmt_bleu' \
  --region_feats_path $FEAT_PATH --image_names ${DATA_PATH} --visual_first true \
  --num_of_regions 36 --reg_enc_bias false --init_dec_from_enc $@
