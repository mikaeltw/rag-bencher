#!/usr/bin/env bash

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "This script must be run as root. Try 'sudo $0'." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive

INSTALL_NVIDIA_DRIVER="${INSTALL_NVIDIA_DRIVER:-1}"
INSTALL_UV="${INSTALL_UV:-0}"

apt-get update
apt-get install -y --no-install-recommends \
  apt-transport-https \
  ca-certificates \
  curl \
  gnupg \
  lsb-release

# Install Docker CE
install -m 0755 -d /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/docker.gpg ]]; then
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
fi
chmod a+r /etc/apt/keyrings/docker.gpg

DOCKER_RELEASE=$(. /etc/os-release && echo "${VERSION_CODENAME}")
cat >/etc/apt/sources.list.d/docker.list <<EOF
deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${DOCKER_RELEASE} stable
EOF

apt-get update
apt-get install -y --no-install-recommends \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin
systemctl enable --now docker

# Install NVIDIA driver if requested
if [[ "${INSTALL_NVIDIA_DRIVER}" == "1" ]]; then
  apt-get update
  apt-get install -y --no-install-recommends nvidia-driver-550
fi

# Install NVIDIA container toolkit
install -m 0755 -d /usr/share/keyrings
distribution=$(. /etc/os-release; echo "${ID}${VERSION_ID}")
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL "https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list" | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y --no-install-recommends nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Install uv (available to default user)
if [[ "${INSTALL_UV}" == "1" ]]; then
  target_user="${SUDO_USER:-ubuntu}"
  if id "${target_user}" &>/dev/null; then
    sudo -u "${target_user}" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
  fi
fi

apt-get clean
rm -rf /var/lib/apt/lists/*

echo "GPU host setup complete."
