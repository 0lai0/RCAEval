#!/usr/bin/env bash

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────
VENV_PATH="venv310"
MINICONDA_DIR="$HOME/miniconda"
CONDA_PY310_ENV="$MINICONDA_DIR/envs/py310base"

# ── Helpers ───────────────────────────────────────────────────────
write_info()  { echo -e "[INFO] $1"; }
write_warn()  { echo -e "\e[33m[WARN] $1\e[0m"; }
write_error() { echo -e "\e[31m[ERROR] $1\e[0m" >&2; }

write_info "Working directory: $(pwd)"

# ── Pre-flight checks ────────────────────────────────────────────
if [ ! -f "requirements.txt" ]; then
    write_error "requirements.txt not found in current directory."
    exit 1
fi

if [ ! -f "requirements_rcd.lock" ]; then
    write_error "requirements_rcd.lock not found in current directory."
    exit 1
fi

# ── Step 1: Ensure Miniconda is installed ─────────────────────────
if [ ! -d "$MINICONDA_DIR" ]; then
    write_info "Installing Miniconda..."
    curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
    rm -f /tmp/miniconda.sh

    # Accept TOS for default channels
    "$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    "$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
fi

# ── Step 2: Ensure Python 3.10 conda env exists (bootstrap only) ──
if [ ! -f "$CONDA_PY310_ENV/bin/python" ]; then
    write_info "Creating Python 3.10 bootstrap environment via conda..."
    "$MINICONDA_DIR/bin/conda" create -n py310base python=3.10 -y
fi

PY310="$CONDA_PY310_ENV/bin/python"
PY_VERSION=$("$PY310" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
write_info "Bootstrap Python version: $PY_VERSION"

# ── Step 3: Create standard venv with Python 3.10 ─────────────────
if [ -d "$VENV_PATH" ]; then
    write_info "Virtual environment '$VENV_PATH' already exists. Skipping creation."
else
    write_info "Creating standard venv '$VENV_PATH' with Python $PY_VERSION..."
    "$PY310" -m venv "$VENV_PATH"
fi

VENV_PYTHON="$VENV_PATH/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    write_error "python binary not found at '$VENV_PYTHON'. Venv creation failed."
    exit 1
fi

write_info "Venv Python: $($VENV_PYTHON --version)"

# ── Step 4: Ensure pip is available ───────────────────────────────
if ! "$VENV_PYTHON" -m pip --version &> /dev/null; then
    write_info "pip not available in venv; running ensurepip..."
    "$VENV_PYTHON" -m ensurepip --upgrade
fi

# ── Step 5: Pin build tooling (tigramite needs older setuptools) ──
write_info "Installing compatible build tooling (pip<24, setuptools<71, Cython)..."
"$VENV_PYTHON" -m pip install --upgrade \
    "pip<24" \
    "setuptools<71" \
    "wheel" \
    "Cython"

# ── Step 6: Pin numpy<2 FIRST (binary incompatibility prevention) ─
write_info "Pinning numpy<2.0 to prevent binary incompatibility..."
"$VENV_PYTHON" -m pip install "numpy<2.0.0"

# ── Step 7: Install tigramite separately (C extension, needs care)
write_info "Installing tigramite (C extension build)..."
"$VENV_PYTHON" -m pip install tigramite==4.2.2.1

# ── Step 8: Install main dependencies ─────────────────────────────
write_info "Installing dependencies from requirements.txt..."
"$VENV_PYTHON" -m pip install -r requirements.txt

# ── Step 9: Install locked RCD dependencies (overrides) ───────────
write_info "Installing locked RCD dependencies from requirements_rcd.lock..."
"$VENV_PYTHON" -m pip install -r requirements_rcd.lock

# ── Step 10: Validate ─────────────────────────────────────────────
write_info "Validating environment (pip check)..."
if ! "$VENV_PYTHON" -m pip check; then
    write_warn "pip check reported issues (often harmless with pinned versions)."
fi

echo ""
echo -e "\e[32m============================================================\e[0m"
echo -e "\e[32mVirtual environment setup completed successfully.\e[0m"
echo ""
echo -e "\e[32mTo activate the environment, run:\e[0m"
echo -e "\e[33m    source ./$VENV_PATH/bin/activate\e[0m"
echo ""
echo -e "\e[32mThen you can run, for example:\e[0m"
echo -e "\e[33m    python main.py --method baro --dataset re2-ob\e[0m"
echo -e "\e[32m============================================================\e[0m"
