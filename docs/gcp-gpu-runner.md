# Running GPU workflows on GCP via Cirun

The GPU job defined in `.github/workflows/_gpu.yml` can now run entirely inside a container that has all rag-bench dependencies pre-installed. This document explains how to build and store:

1. A GCE VM image that already contains Docker and the NVIDIA Container Toolkit so Cirun can boot a runner quickly.
2. A GPU-enabled container image that bundles rag-bench plus its dev tooling so `_gpu.yml` can run the existing tox workflow inside the container.

## Prerequisites

- `gcloud` CLI authenticated against the target project with permissions to create instances, images and Artifact Registry repositories.
- `docker` CLI authenticated against GCP Artifact Registry.
- GPU quota in the selected region/zone (for example, one `nvidia-tesla-t4`).
- Cirun account connected to your GitHub repo.

## Build the GPU host VM image

1. Launch and configure an instance that installs Docker CE and the NVIDIA container toolkit by running:

   ```bash
   export GCP_PROJECT="my-project"
   export GCP_ZONE="us-central1-b"
   ./scripts/gcp/build_gpu_base_image.sh
   ```

   The helper script provisions a temporary GPU VM based on Ubuntu 22.04, copies `scripts/gcp/install_gpu_host.sh` to it, runs the installer (Docker, NVIDIA driver 550, NVIDIA container toolkit) and saves the stopped disk as a reusable image family (`rag-bench-gpu-host` by default). The builder VM defaults to a smaller machine/boot disk and preemptible pricing; override via `MACHINE_TYPE`, `BOOT_DISK_SIZE` or `PREEMPTIBLE`. Set `IMAGE_NAME`, `IMAGE_FAMILY`, `GPU_TYPE`, etc. to override the other defaults.

2. Verify the image is stored in your project:

   ```bash
   gcloud compute images list --filter="name~'rag-bench-gpu-host'"
   ```

3. (Optional) Export the image to Cloud Storage if you need to keep an archive or share it across projects:

   ```bash
   gcloud compute images export \
     --image rag-bench-gpu-host \
     --destination-uri gs://my-bucket/rag-bench-gpu-host.tar.gz
   ```

   You can later re-import the archive into another project with `gcloud compute images import`.

## Use the host image from Cirun

Add a Cirun job that references the image family to `.cirun.yml` (create the file if it does not exist):

```yaml
gpu-runner:
  labels: [self-hosted, linux, x64, gpu]
  provider:
    name: gcp
    project: gpu-test-runners
    region: us-central1
    zone: us-central1-b
  machine:
    type: n1-standard-4
    gpu:
      type: nvidia-tesla-t4
      count: 1
    image:
      family: rag-bench-gpu-host
      project: gpu-test-runners
    serviceAccount: gpu-test-runners-sa@gpu-test-runners.iam.gserviceaccount.com
    serviceAccountScopes: [cloud-platform]
    username: ci-gpu-runner
  setup:
    - name: Authenticate Docker to Artifact Registry and pre-pull image
      run: |
        set -euo pipefail
        HOST="${GCP_ARTIFACT_REGION:-us-central1}-docker.pkg.dev"
        PROJECT="${GCP_PROJECT:-gpu-test-runners}"
        IMAGE_REF="${HOST}/${PROJECT}/rag-bench/rag-bench-gpu-tests:latest"
        gcloud auth configure-docker ${HOST} --quiet
        # TOKEN=$(curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" | awk -F'"' '/access_token/ {print $4}')
        # echo "${TOKEN}" | docker login -u oauth2accesstoken --password-stdin "https://${HOST}"
        docker pull "${IMAGE_REF}"
```

Cirun will now boot runners from your preconfigured image, so the VM already has Docker plus the correct NVIDIA runtime when the GitHub Actions job starts. Adjust the resource labels to match `.github/workflows/_gpu.yml` (`[self-hosted, linux, x64, gpu]`).

## Build and publish the rag-bench GPU test container

1. Build the container that runs the tox workflow:

   ```bash
   GCP_PROJECT="my-project" GCP_ARTIFACT_REGION="us-central1" ./scripts/docker/build_gpu_test_image.sh
   ```

   By default it builds with a split CUDA base to keep the final image smaller: `nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04` for the build stage and `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04` for the runtime stage. Override via `CUDA_IMAGE_DEVEL=...` and `CUDA_IMAGE_RUNTIME=...`. The Dockerfile at `docker/gpu-tests.Dockerfile` installs uv, copies the repo and relies on the runtime entrypoint to execute `make setup && make sync && make test-all-gpu` (the same sequence used in `_gpu.yml`). Dependencies download during `make sync` inside the container and land in the mounted cache directories, so subsequent workflow runs reuse them.

2. Push to GCP Artifact Registry:

   ```bash
   gcloud auth configure-docker us-central1-docker.pkg.dev
   IMAGE_REPO="us-central1-docker.pkg.dev/my-project/rag-bench/rag-bench-gpu-tests" PUSH=1 ./scripts/docker/build_gpu_test_image.sh
   ```

   Use the Artifact Registry region that matches your runner zone’s region (for example: zone `us-central1-b` ⇒ host `us-central1-docker.pkg.dev`). Keeping the registry and VM in the same region minimizes egress time/cost.
   For faster pulls/extraction, attach an NVMe local SSD to the runner VM and let the host image place Docker’s data-root on it automatically (see below).

## Running the workflow inside the container

The `_gpu.yml` workflow now calls `scripts/docker/run_gpu_tests.sh`, which:

- Pulls the configured image (default is `${GCP_ARTIFACT_REGION:-us}-docker.pkg.dev/${GCP_PROJECT}/rag-bench/rag-bench-gpu-tests:latest` via workflow/repo vars).
- Runs it with `--gpus all`.
- Mounts the working directory and cache folders (`~/.cache/uv`, `~/.cache/huggingface`, `~/.cache/torch`) so existing GitHub Action cache steps still apply.

You can also use the script locally:

```bash
GCP_PROJECT="my-project" GCP_ARTIFACT_REGION="us-central1" ./scripts/docker/run_gpu_tests.sh
```

Override the command to run ad-hoc checks (for example, `./scripts/docker/run_gpu_tests.sh bash -lc "pytest tests/gpu -k cache"`).

## Summary of storage locations

- **Host VM image** – lives as a Compute Engine image (and optional Cloud Storage export) inside your GCP project. Cirun references it via the image family.
- **GPU test container** – stored in GCP Artifact Registry in the same region as your GPU runners for low-latency pulls. Use `IMAGE_REPO`/`GCP_ARTIFACT_REGION` to control the exact location.

With these scripts plus the updated workflow, Cirun can boot a GPU VM that immediately runs the rag-bench GPU tox suite inside the prepared container while reusing GitHub Action caches for dependencies and model weights. The host image includes a boot-time service that, if it detects a local SSD (e.g., GCE NVMe local-ssd), will format/mount it at `/mnt/local-ssd` and set Docker’s `data-root` there so image extraction uses NVMe bandwidth. Attach a local SSD to runner VMs (and ensure the service account has Artifact Registry pull access) for the best pull times.
