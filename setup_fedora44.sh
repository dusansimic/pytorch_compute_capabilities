#!/usr/bin/env bash
#
# Install everything needed to run the pytorch_compute_capabilities scripts on
# a clean Fedora 44 machine (e.g. a headless VPS). No GPU or NVIDIA driver is
# required: cuobjdump is a static fat-binary inspector, not a runtime tool.
#
# What it installs:
#   - system packages: git, tar, bzip2 (archive handling), curl if missing
#   - uv               -> runs the scripts; resolves their inline Python deps
#   - Miniforge (conda)-> only used to pull a self-contained cuobjdump
#   - cuobjdump        -> from the nvidia conda channel (cuda-cuobjdump)
#
# The Python dependencies of the scripts themselves are NOT installed here:
# they are declared inline (PEP 723) and `uv run` installs them on demand.
#
# Usage:
#   bash setup_fedora44.sh
#
set -euo pipefail

log() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

# --- 1. System packages -----------------------------------------------------
log "Installing system packages..."
sudo dnf install -y git tar bzip2
# curl-minimal ships on Fedora and provides /usr/bin/curl; only install the
# full curl if no curl is present at all.
command -v curl >/dev/null 2>&1 || sudo dnf install -y curl

# --- 2. uv ------------------------------------------------------------------
if command -v uv >/dev/null 2>&1; then
  log "uv already installed: $(uv --version)"
else
  log "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# --- 3. Miniforge (conda) ---------------------------------------------------
CONDA_DIR="$HOME/miniforge3"
if [ -x "$CONDA_DIR/bin/conda" ]; then
  log "Miniforge already present at $CONDA_DIR"
else
  log "Installing Miniforge to $CONDA_DIR..."
  curl -LsSf -o /tmp/miniforge.sh \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
  rm -f /tmp/miniforge.sh
fi

# --- 4. cuobjdump via the nvidia conda channel ------------------------------
if [ -x "$CONDA_DIR/envs/cuobjdump/bin/cuobjdump" ]; then
  log "cuobjdump env already exists"
else
  log "Creating conda env 'cuobjdump' with nvidia::cuda-cuobjdump..."
  "$CONDA_DIR/bin/conda" create -y -n cuobjdump -c nvidia cuda-cuobjdump
fi

# --- 5. Put cuobjdump on PATH -----------------------------------------------
mkdir -p "$HOME/.local/bin"
ln -sf "$CONDA_DIR/envs/cuobjdump/bin/cuobjdump" "$HOME/.local/bin/cuobjdump"

# --- 6. Verify --------------------------------------------------------------
log "Verifying installation..."
export PATH="$HOME/.local/bin:$PATH"
uv --version
cuobjdump --version | head -1

cat <<'EOF'

============================================================
Setup complete.

Ensure ~/.local/bin is on your PATH (new login shells usually
add it automatically). For the current shell:

    export PATH="$HOME/.local/bin:$PATH"

Run the scripts:

    # conda channels: pytorch | conda-forge | anaconda
    uv run pytorch_compute_capabilities.py --channel pytorch
    uv run pytorch_compute_capabilities.py --channel conda-forge

    # PyPI wheels (-y skips the download-size confirmation)
    uv run pytorch_compute_capabilities_pip.py -y

    # PyTorch download index (default: cu118 cu121 cu124 cu126 cu128)
    uv run pytorch_compute_capabilities_download.py -y

These are long, bandwidth-heavy jobs. On a VPS run them under
tmux so they survive disconnects:

    tmux new -s pcc
    uv run pytorch_compute_capabilities.py --channel conda-forge
    # detach: Ctrl-b then d   |   reattach: tmux attach -t pcc

For long-running scripts, set a disk-backed temp directory to avoid
tmpfs exhaustion (especially on Hetzner VMs where /tmp is small):

    uv run pytorch_compute_capabilities_pip.py --tmpdir /var/tmp -y
    uv run pytorch_compute_capabilities_download.py --tmpdir /var/tmp -y

All scripts are resumable: progress is cached (cache/, cache_pip/,
cache_download/) and re-running skips work that is already done.
============================================================
EOF
