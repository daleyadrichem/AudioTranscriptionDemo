#!/usr/bin/env bash

set -euo pipefail

MODEL_NAME="${1:-vosk-model-small-en-us-0.15}"
MODEL_URL="https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"
MODELS_DIR="${MODELS_DIR:-models}"

echo "Requested Vosk model: ${MODEL_NAME}"
echo "Target directory: ${MODELS_DIR}"

mkdir -p "${MODELS_DIR}"

if [ -d "${MODELS_DIR}/${MODEL_NAME}" ]; then
    echo "Model already exists at ${MODELS_DIR}/${MODEL_NAME}"
    echo "Skipping download."
    exit 0
fi

TMP_DIR="$(mktemp -d)"
ZIP_PATH="${TMP_DIR}/${MODEL_NAME}.zip"

echo "Downloading model..."
curl -L "${MODEL_URL}" -o "${ZIP_PATH}"

echo "Extracting model..."
unzip -q "${ZIP_PATH}" -d "${TMP_DIR}"

echo "Moving model to ${MODELS_DIR}"
mv "${TMP_DIR}/${MODEL_NAME}" "${MODELS_DIR}/"

echo "Cleaning temporary files"
rm -rf "${TMP_DIR}"

echo "Model installed at:"
echo "${MODELS_DIR}/${MODEL_NAME}"