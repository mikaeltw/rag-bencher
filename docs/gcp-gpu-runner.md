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
    project: my-project
    region: us-central1
    zone: us-central1-b
  machine:
    type: n1-standard-8
    gpu:
      type: nvidia-tesla-t4
      count: 1
    image:
      family: rag-bench-gpu-host
      project: my-project
  setup:
    # Pre-pull the GPU test container (optional but speeds up the job)
    - docker pull us-central1-docker.pkg.dev/my-project/rag-bench/rag-bench-gpu-tests:latest
```

Cirun will now boot runners from your preconfigured image, so the VM already has Docker plus the correct NVIDIA runtime when the GitHub Actions job starts. Adjust the resource labels to match `.github/workflows/_gpu.yml` (`[self-hosted, linux, x64, gpu]`).

## Build and publish the rag-bench GPU test container

1. Build the container that runs the tox workflow:

   ```bash
   GCP_PROJECT="my-project" GCP_ARTIFACT_REGION="us-central1" ./scripts/docker/build_gpu_test_image.sh
   ```

   By default it uses the `nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04` base image (configurable via `CUDA_IMAGE=...`). The Dockerfile at `docker/gpu-tests.Dockerfile` installs uv, copies the repo and embeds `/usr/local/bin/run-gpu-tests.sh`, which executes the same `make setup && make sync && make test-all-gpu` sequence used in `_gpu.yml`. Dependencies download during `make sync` inside the container and land in the mounted cache directories, so subsequent workflow runs reuse them.

2. Push to GCP Artifact Registry:

   ```bash
   gcloud auth configure-docker us-central1-docker.pkg.dev
   IMAGE_REPO="us-central1-docker.pkg.dev/my-project/rag-bench/rag-bench-gpu-tests" PUSH=1 ./scripts/docker/build_gpu_test_image.sh
   ```

   Use the Artifact Registry region that matches your runner zone’s region (for example: zone `us-central1-b` ⇒ host `us-central1-docker.pkg.dev`). Keeping the registry and VM in the same region minimizes egress time/cost.

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

With these scripts plus the updated workflow, Cirun can boot a GPU VM that immediately runs the rag-bench GPU tox suite inside the prepared container while reusing GitHub Action caches for dependencies and model weights.
