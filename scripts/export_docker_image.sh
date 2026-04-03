#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="${ROOT_DIR}/dist"
IMAGE_NAME="${1:-multimodal-graph-rag-agent}"
IMAGE_TAG="${2:-offline}"
ARCHIVE_BASENAME="${3:-${IMAGE_NAME//\//_}-${IMAGE_TAG}}"
SPLIT_SIZE="${SPLIT_SIZE:-1900m}"
DOCKER_DESKTOP_BIN="/Applications/Docker.app/Contents/Resources/bin"
PROJECT_DOCKER_CONFIG="${ROOT_DIR}/.docker-buildx-config"

if [[ -d "${DOCKER_DESKTOP_BIN}" ]]; then
    export PATH="${DOCKER_DESKTOP_BIN}:${PATH}"
fi

if [[ -d "${PROJECT_DOCKER_CONFIG}" ]]; then
    export DOCKER_CONFIG="${PROJECT_DOCKER_CONFIG}"
fi

mkdir -p "${DIST_DIR}"

ARCHIVE_PATH="${DIST_DIR}/${ARCHIVE_BASENAME}.tar.gz"

docker save "${IMAGE_NAME}:${IMAGE_TAG}" | gzip > "${ARCHIVE_PATH}"

if command -v split >/dev/null 2>&1; then
    rm -f "${ARCHIVE_PATH}.part-"*
    split -b "${SPLIT_SIZE}" -d "${ARCHIVE_PATH}" "${ARCHIVE_PATH}.part-"
    echo "镜像已导出: ${ARCHIVE_PATH}"
    echo "若需上传 GitHub Release，可使用分片文件: ${ARCHIVE_PATH}.part-*"
else
    echo "镜像已导出: ${ARCHIVE_PATH}"
    echo "当前系统没有 split，未生成分片文件。"
fi
