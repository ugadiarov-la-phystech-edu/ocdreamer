SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/.env"

seed=0
script=train_eval
model=ocdreamer
task=langroom
device=0

export COMET_API_KEY
export COMET_PROJECT_NAME=compare_ocdreamer_dynalang
export COMET_EXPERIMENT_NAME=model-${model}_task-${task}_seed-${seed}
export COMET_RUN_ID=$COMET_RUN_ID_LANGROOM

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
    --run.log_every 1000 
# --run.from_checkpoint 
