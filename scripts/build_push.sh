#!/usr/bin/env bash
# Build the pod image from the frozen lock file and push it with an immutable tag.
#   REGISTRY=docker.io/<you> scripts/build_push.sh v1
# Needs Docker with buildx (linux/amd64). No GPU needed to build.
set -euo pipefail
TAG="${1:?usage: build_push.sh <tag e.g. v1>   (never 'latest')}"
[ "$TAG" != "latest" ] || { echo "refusing to tag :latest"; exit 1; }
REGISTRY="${REGISTRY:?set REGISTRY, e.g. docker.io/yourname or ghcr.io/yourname}"
IMAGE="${REGISTRY}/imgedit:${TAG}"
cd "$(dirname "$0")/../server"

if head -1 requirements.lock.txt | grep -q PROVISIONAL; then
  echo "WARNING: requirements.lock.txt is still the provisional lock. Run Phase 0 first for a reproducible build." >&2
fi

docker buildx build --platform linux/amd64 -t "$IMAGE" --push .
echo
echo "pushed $IMAGE"
docker buildx imagetools inspect "$IMAGE" | grep -m1 Digest || true
echo "Record the tag + digest in NOTES.md. On RunPod: Container Image = $IMAGE, Expose HTTP Ports = 8000,"
echo "env API_TOKEN=<random>, HF_HOME=/workspace/hf, container disk 100GB."
