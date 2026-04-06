#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

seed=0
script=train_eval
model=ocdreamer
task=langroom
device=5

export COMET_API_KEY
export COMET_PROJECT_NAME=langroom_flatten_slots_ocdreamer
export COMET_EXPERIMENT_NAME=model-${model}_task-${task}_seed-${seed}_flatten_slots
#export COMET_RUN_ID=$COMET_RUN_ID_LANGROOM

SLOT_EXTRACTOR_TYPE="slotcontrast"             
SLOT_EXTRACTOR_CONFIG="$SLOT_EXTRACTOR_CONFIG_LANGROOM"
SLOT_EXTRACTOR_CHECKPOINT="$SLOT_EXTRACTOR_CHECKPOINT_LANGROOM"
SLOT_EXTRACTOR_DEVICE="cuda:0"                # device for slot extractor
BACKBONE_INPUT_SIZE=336                    

export CUDA_VISIBLE_DEVICES=$device; python dreamerv3/main.py  --configs dynalang langroom\
    --script ${script} \
    --seed ${seed} \
    --logdir data/${model}/${COMET_EXPERIMENT_NAME} \
    --logger.outputs comet \
    --jax.mem_fraction 0.95 \
    --jax.platform cuda,cpu \
    --jax.profiler False \
    --run.eval_eps 50 \
    --agent.dec.typ resnet \
    --agent.enc.typ resnet \
    --run.report_every 25e4 \
    --run.log_every 1000 \
    \
    --agent.batch_env.use_slot_extractor True \
    --agent.batch_env.use_flatten_slots True \
    --agent.batch_env.batch_slot_extractor_env.slot_extractor.typ "$SLOT_EXTRACTOR_TYPE" \
    --agent.batch_env.batch_slot_extractor_env.slot_extractor.config_path "$SLOT_EXTRACTOR_CONFIG" \
    --agent.batch_env.batch_slot_extractor_env.slot_extractor.checkpoint_path "$SLOT_EXTRACTOR_CHECKPOINT" \
    --agent.batch_env.batch_slot_extractor_env.slot_extractor.device "$SLOT_EXTRACTOR_DEVICE" \
    --agent.batch_env.batch_slot_extractor_env.slot_extractor.backbone_input_size "$BACKBONE_INPUT_SIZE" \
    \
    --agent.enc.resnet.vec_keys 'text$|flatten_slots$' \
    --agent.enc.resnet.img_keys '$^' \
    --agent.enc.resnet.pass_keys 'flatten_slots' \
    --agent.dec.resnet.vec_keys 'text$|flatten_slots$' \
    --agent.dec.resnet.img_keys '$^' \
    --agent.dec.resnet.vec_dists 'text:onehot,flatten_slots:mse' \
    --agent.batch_env.batch_slot_extractor_env.use_previous_slots True