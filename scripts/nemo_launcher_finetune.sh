#!/usr/bin/env bash
set -euo pipefail

NEMO_FRAMEWORK_LAUNCHER_DIR=${NEMO_FRAMEWORK_LAUNCHER_DIR:-"/opt/NeMo-Framework-Launcher"}
if [[ ! -d "${NEMO_FRAMEWORK_LAUNCHER_DIR}" ]]; then
  echo "NeMo Framework Launcher not found at ${NEMO_FRAMEWORK_LAUNCHER_DIR}" >&2
  exit 1
fi

DATA_DIR=${DATA_DIR:?DATA_DIR is required}
OUTPUT_DIR=${OUTPUT_DIR:?OUTPUT_DIR is required}
MODEL_NAME=${MODEL_NAME:?MODEL_NAME is required}

TRAIN_DS="${DATA_DIR}/train.jsonl"
VALID_DS="${DATA_DIR}/val.jsonl"

if [[ ! -f "${TRAIN_DS}" || ! -f "${VALID_DS}" ]]; then
  echo "Missing train.jsonl or val.jsonl in ${DATA_DIR}" >&2
  exit 1
fi

TUNING_METHOD=${TUNING_METHOD:-lora}
if [[ "${TUNING_METHOD}" == "sft" ]]; then
  PEFT_SCHEME="null"
else
  PEFT_SCHEME="lora"
fi

TRAINER_DEVICES=${TRAINER_DEVICES:-1}
TRAINER_NODES=${TRAINER_NODES:-1}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
PIPELINE_PARALLEL_SIZE=${PIPELINE_PARALLEL_SIZE:-1}
MICRO_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
GRAD_ACC=${GRADIENT_ACCUMULATION_STEPS:-1}
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * GRAD_ACC))
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-1024}

RESULTS_DIR="${OUTPUT_DIR}/nemo_results"
EXTRA_ARGS=${NEMO_EXTRA_ARGS:-""}

python3 "${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py" \
  peft=nemotron/sft \
  stages=[peft] \
  launcher_scripts_path="${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts" \
  base_results_dir="${RESULTS_DIR}" \
  peft.run.name="nemotron_mini_4b_${TUNING_METHOD}" \
  peft.trainer.devices="${TRAINER_DEVICES}" \
  peft.trainer.num_nodes="${TRAINER_NODES}" \
  peft.model.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE}" \
  peft.model.pipeline_model_parallel_size="${PIPELINE_PARALLEL_SIZE}" \
  peft.model.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  peft.model.micro_batch_size="${MICRO_BATCH_SIZE}" \
  peft.model.restore_from_path="${MODEL_NAME}" \
  peft.model.peft.peft_scheme="${PEFT_SCHEME}" \
  peft.model.data.train_ds.file_names="${TRAIN_DS}" \
  peft.model.data.validation_ds.file_names="${VALID_DS}" \
  peft.model.data.train_ds.max_seq_length="${MAX_SEQ_LENGTH}" \
  peft.model.data.train_ds.prompt_template="{input}{output}" \
  peft.model.data.train_ds.truncation_field="input" \
  ${EXTRA_ARGS}
