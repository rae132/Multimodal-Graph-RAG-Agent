#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "用法:"
    echo "  $0 /path/to/image.tar.gz"
    echo "  $0 /path/to/image.tar.gz.part-00 /path/to/image.tar.gz.part-01 ..."
    exit 1
fi

if [[ $# -eq 1 ]]; then
    gzip -dc "$1" | docker load
    exit 0
fi

TMP_ARCHIVE="$(mktemp /tmp/docker-image-XXXXXX.tar.gz)"
cat "$@" > "${TMP_ARCHIVE}"
gzip -dc "${TMP_ARCHIVE}" | docker load
rm -f "${TMP_ARCHIVE}"
