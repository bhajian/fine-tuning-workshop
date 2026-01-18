#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NEMO_FRAMEWORK_LAUNCHER_DIR=${NEMO_FRAMEWORK_LAUNCHER_DIR:-"/opt/NeMo-Framework-Launcher"}
NEMO_CODE_PATH=${NEMO_CODE_PATH:-"/opt/NeMo"}
MEGATRON_LM_PATH=${MEGATRON_LM_PATH:-"/opt/megatron-lm"}
NEMO_VENV=${NEMO_VENV:-""}
NEMO_PYTHON=${NEMO_PYTHON:-"python3"}
NEMO_GPT_FINETUNING_SCRIPT=${NEMO_GPT_FINETUNING_SCRIPT:-""}
if [[ ! -d "${NEMO_FRAMEWORK_LAUNCHER_DIR}" ]]; then
  echo "NeMo Framework Launcher not found at ${NEMO_FRAMEWORK_LAUNCHER_DIR}" >&2
  exit 1
fi

if [[ -n "${NEMO_VENV}" ]]; then
  if [[ ! -x "${NEMO_VENV}/bin/python" ]]; then
    echo "NEMO_VENV is set but no python found at ${NEMO_VENV}/bin/python" >&2
    exit 1
  fi
  NEMO_PYTHON="${NEMO_VENV}/bin/python"
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
NEMO_LAUNCHER_CLUSTER=${NEMO_LAUNCHER_CLUSTER:-""}

if [[ ! -d "/opt/NeMo" ]]; then
  if [[ "${NEMO_CODE_PATH}" != "/opt/NeMo" && -d "${NEMO_CODE_PATH}" ]]; then
    if [[ -w "/opt" ]]; then
      ln -s "${NEMO_CODE_PATH}" /opt/NeMo
    else
      echo "/opt/NeMo is missing. Create a symlink with:" >&2
      echo "  sudo ln -s ${NEMO_CODE_PATH} /opt/NeMo" >&2
      exit 1
    fi
  else
    echo "/opt/NeMo is missing. Install NeMo or set NEMO_CODE_PATH to your clone." >&2
    echo "  sudo git clone https://github.com/NVIDIA/NeMo.git /opt/NeMo" >&2
    exit 1
  fi
fi

if [[ ! -d "/opt/megatron-lm" && "${MEGATRON_LM_PATH}" != "/opt/megatron-lm" && -d "${MEGATRON_LM_PATH}" ]]; then
  if [[ -w "/opt" ]]; then
    ln -s "${MEGATRON_LM_PATH}" /opt/megatron-lm
  else
    echo "Warning: /opt/megatron-lm is missing; git logs will be skipped." >&2
  fi
fi

EXPECTED_GPT_FINETUNE="/opt/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py"
ALT_GPT_FINETUNE="/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_finetuning.py"

if [[ ! -f "${EXPECTED_GPT_FINETUNE}" ]]; then
  if [[ -f "${ALT_GPT_FINETUNE}" ]]; then
    mkdir -p "$(dirname "${EXPECTED_GPT_FINETUNE}")"
    if ! ln -sf "${ALT_GPT_FINETUNE}" "${EXPECTED_GPT_FINETUNE}"; then
      echo "Found ${ALT_GPT_FINETUNE} but could not create a symlink at ${EXPECTED_GPT_FINETUNE}." >&2
      echo "Try: sudo ln -sf ${ALT_GPT_FINETUNE} ${EXPECTED_GPT_FINETUNE}" >&2
      exit 1
    fi
  elif [[ -n "${NEMO_GPT_FINETUNING_SCRIPT}" ]]; then
    if [[ ! -f "${NEMO_GPT_FINETUNING_SCRIPT}" ]]; then
      echo "NEMO_GPT_FINETUNING_SCRIPT is set but not found: ${NEMO_GPT_FINETUNING_SCRIPT}" >&2
      exit 1
    fi
    mkdir -p "$(dirname "${EXPECTED_GPT_FINETUNE}")"
    if ! ln -sf "${NEMO_GPT_FINETUNING_SCRIPT}" "${EXPECTED_GPT_FINETUNE}"; then
      echo "Could not create a symlink at ${EXPECTED_GPT_FINETUNE}." >&2
      echo "Try: sudo ln -sf ${NEMO_GPT_FINETUNING_SCRIPT} ${EXPECTED_GPT_FINETUNE}" >&2
      exit 1
    fi
  else
    echo "NeMo training script not found under /opt/NeMo." >&2
    echo "Expected: ${EXPECTED_GPT_FINETUNE}" >&2
    echo "If your NeMo repo uses a different path, set NEMO_GPT_FINETUNING_SCRIPT to it." >&2
    exit 1
  fi
fi

if [[ -n "${NEMO_VENV}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} env_vars.PATH=${NEMO_VENV}/bin:$PATH"
fi

if [[ -z "${NEMO_LAUNCHER_CLUSTER}" ]]; then
  if ! command -v srun >/dev/null 2>&1; then
    NEMO_LAUNCHER_CLUSTER="interactive"
  fi
fi

if [[ "${NEMO_LAUNCHER_CLUSTER}" == "interactive" ]]; then
  LOCAL_CONF_DIR="${SCRIPT_DIR}/nemo_conf"
  if [[ ! -d "${LOCAL_CONF_DIR}" ]]; then
    echo "Local NeMo config dir not found at ${LOCAL_CONF_DIR}" >&2
    exit 1
  fi
  EXTRA_ARGS="${EXTRA_ARGS} hydra.searchpath=[file://${LOCAL_CONF_DIR}] cluster=interactive cluster_type=interactive"
fi

"${NEMO_PYTHON}" "${NEMO_FRAMEWORK_LAUNCHER_DIR}/launcher_scripts/main.py" \
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
  peft.model.data.train_ds.prompt_template="'{input}{output}'" \
  peft.model.data.train_ds.truncation_field="input" \
  ${EXTRA_ARGS}
