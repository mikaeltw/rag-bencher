#!/usr/bin/env bash

set -euo pipefail

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to build the GPU test image." >&2
  exit 1
fi

# Default to a regional Artifact Registry repo close to the GPU VMs.
if [[ -z "${IMAGE_REPO:-}" ]]; then
  : "${GCP_PROJECT:?Set GCP_PROJECT to your GCP project id or override IMAGE_REPO}"
  GCP_ARTIFACT_REGION="${GCP_ARTIFACT_REGION:-us}"
  IMAGE_REPO="${GCP_ARTIFACT_REGION}-docker.pkg.dev/${GCP_PROJECT}/rag-bench/rag-bench-gpu-tests"
fi

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)"
IMAGE_TAG="${IMAGE_TAG:-${GIT_SHA}}"
IMAGE_REF="${IMAGE_REPO}:${IMAGE_TAG}"
PUSH="${PUSH:-0}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04}"

echo "Building GPU test image ${IMAGE_REF} (base=${CUDA_IMAGE})"

docker build \
  -f docker/gpu-tests.Dockerfile \
  --build-arg "CUDA_IMAGE=${CUDA_IMAGE}" \
  -t "${IMAGE_REF}" \
  .

if [[ "${PUSH}" == "1" ]]; then
  echo "Pushing ${IMAGE_REF}"
  docker push "${IMAGE_REF}"
fi

cat <<EOF
Built image: ${IMAGE_REF}
To run GPU tests locally:
  IMAGE_REF="${IMAGE_REF}" ./scripts/docker/run_gpu_tests.sh
EOF
