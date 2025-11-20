ARG CUDA_IMAGE=nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/opt/uv-cache \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/rag-bench/.venv \
    PATH="/root/.local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      curl \
      ca-certificates \
      pkg-config \
      python3 \
      python3-venv \
      python3-pip \
      python3-dev \
      make \
      jq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

FROM base AS deps

COPY pyproject.toml uv.lock README.md LICENSE Makefile ./
COPY src ./src

RUN make setup && make sync && \
    uv python install 3.13 && \
    uv python install 3.14 && \
    uv cache prune --ci

FROM base AS runtime

COPY --from=deps /root/.local /root/.local
COPY --from=deps /opt/rag-bench/.venv /opt/rag-bench/.venv
COPY --from=deps /opt/uv-cache /opt/uv-cache
COPY --from=deps /workspace/pyproject.toml pyproject.toml
COPY --from=deps /workspace/uv.lock uv.lock
COPY --from=deps /workspace/README.md README.md
COPY --from=deps /workspace/LICENSE LICENSE
COPY --from=deps /workspace/Makefile Makefile

COPY src tests configs scripts docker docs examples ./
COPY bench.py bench_many.py run.py CHANGELOG.md CONTRIBUTING.md RELEASE.md SECURITY.md MANIFEST.in CODE_OF_CONDUCT.md .dockerignore ./

RUN printf '#!/usr/bin/env bash\nset -euo pipefail\nmake setup && make sync && make test-all-gpu\n' \
    > /usr/local/bin/run-gpu-tests.sh && \
    chmod +x /usr/local/bin/run-gpu-tests.sh

ENTRYPOINT ["/bin/bash"]
