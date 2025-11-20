#!/usr/bin/env bash

set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI is required but not found in PATH." >&2
  exit 1
fi

: "${GCP_PROJECT:?Set GCP_PROJECT to the target project id}"
: "${GCP_ZONE:?Set GCP_ZONE to the zone to use}"

IMAGE_FAMILY="${IMAGE_FAMILY:-rag-bench-gpu-host}"
IMAGE_NAME="${IMAGE_NAME:-${IMAGE_FAMILY}-$(date +%Y%m%d-%H%M%S)}"
INSTANCE_NAME="${INSTANCE_NAME:-${IMAGE_NAME}-builder}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-50}"
SOURCE_IMAGE_FAMILY="${SOURCE_IMAGE_FAMILY:-ubuntu-2204-lts}"
SOURCE_IMAGE_PROJECT="${SOURCE_IMAGE_PROJECT:-ubuntu-os-cloud}"
NETWORK="${NETWORK:-default}"
SUBNET="${SUBNET:-default}"
REGION="${REGION:-${GCP_ZONE%-*}}"
INSTALL_SCRIPT="${INSTALL_SCRIPT:-scripts/gcp/install_gpu_host.sh}"
PREEMPTIBLE="${PREEMPTIBLE:-true}"

echo "Using:"
echo "  Project:        ${GCP_PROJECT}"
echo "  Zone:           ${GCP_ZONE}"
echo "  Instance:       ${INSTANCE_NAME}"
echo "  Image name:     ${IMAGE_NAME}"
echo "  Machine type:   ${MACHINE_TYPE}"
echo "  GPU:            ${GPU_TYPE} x${GPU_COUNT}"
echo "  Boot disk (GB): ${BOOT_DISK_SIZE}"
echo "  Preemptible:    ${PREEMPTIBLE}"

gcloud config set project "${GCP_PROJECT}" >/dev/null

gcloud compute instances create "${INSTANCE_NAME}" \
  --zone "${GCP_ZONE}" \
  --machine-type "${MACHINE_TYPE}" \
  --accelerator "type=${GPU_TYPE},count=${GPU_COUNT}" \
  --maintenance-policy TERMINATE \
  --provisioning-model STANDARD \
  --boot-disk-type=pd-ssd \
  --boot-disk-size "${BOOT_DISK_SIZE}"GB \
  --image-family "${SOURCE_IMAGE_FAMILY}" \
  --image-project "${SOURCE_IMAGE_PROJECT}" \
  --scopes cloud-platform \
  --network "${NETWORK}" \
  --subnet "${SUBNET}" \
  --tags rag-bench-gpu-builder \
  $(if [[ "${PREEMPTIBLE}" == "true" || "${PREEMPTIBLE}" == "1" ]]; then echo "--preemptible"; fi)

gcloud compute scp \
  --zone "${GCP_ZONE}" \
  "${INSTALL_SCRIPT}" \
  "${INSTANCE_NAME}:~/install_gpu_host.sh"

gcloud compute ssh "${INSTANCE_NAME}" \
  --zone "${GCP_ZONE}" \
  --command "chmod +x ~/install_gpu_host.sh && sudo ~/install_gpu_host.sh"

gcloud compute instances stop "${INSTANCE_NAME}" --zone "${GCP_ZONE}"

gcloud compute images create "${IMAGE_NAME}" \
  --source-disk "${INSTANCE_NAME}" \
  --source-disk-zone "${GCP_ZONE}" \
  --family "${IMAGE_FAMILY}"

gcloud compute instances delete "${INSTANCE_NAME}" --zone "${GCP_ZONE}" --quiet

echo "Image ${IMAGE_NAME} (family: ${IMAGE_FAMILY}) is ready in project ${GCP_PROJECT}."
