#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${1:-multimodal-graph-rag-agent}"
IMAGE_TAG="${2:-offline}"
TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"
DOCKER_DESKTOP_BIN="/Applications/Docker.app/Contents/Resources/bin"
PROJECT_DOCKER_CONFIG="${ROOT_DIR}/.docker-buildx-config"

cd "${ROOT_DIR}"

if [[ -d "${DOCKER_DESKTOP_BIN}" ]]; then
    export PATH="${DOCKER_DESKTOP_BIN}:${PATH}"
fi

mkdir -p "${PROJECT_DOCKER_CONFIG}"
if [[ ! -f "${PROJECT_DOCKER_CONFIG}/config.json" ]]; then
    printf '{\n  "auths": {}\n}\n' > "${PROJECT_DOCKER_CONFIG}/config.json"
fi
export DOCKER_CONFIG="${PROJECT_DOCKER_CONFIG}"
docker build \
    --platform "${TARGET_PLATFORM}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .
