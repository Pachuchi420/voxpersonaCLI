#!/usr/bin/env bash
# Installs backend dependencies (BackService.py + friends) into a Python 3.12
# venv. Run this instead of `pip install -r requirements.txt` directly --
# requirements.txt documents the exact verified version pins, but a single
# joint `pip install -r requirements.txt` fails with ResolutionImpossible:
# parakeet_mlx declares numpy>=2.2.5, chatterbox-tts declares numpy<2.0.0,
# and pip's resolver rejects that combination even though numpy 2.5.2 (what
# mlx-vlm itself pulls in) has been verified to work fine with chatterbox-tts
# at runtime despite its stale declared ceiling. Installing in separate
# `pip install` calls, as below, avoids pip re-validating already-installed
# packages' constraints against each new install and lets the two coexist.
#
# Requires Python 3.12 specifically -- Python 3.14 (the newest Homebrew
# default at the time this was written) is too new for kokoro's
# misaki[en] -> spacy -> thinc -> blis chain to have prebuilt wheels for;
# building blis from source fails there with a Cython/GIL error.
set -euo pipefail

VENV_DIR="${1:-$(dirname "$0")/../voxpersona-venv}"
PYTHON312="$(command -v python3.12 || echo /opt/homebrew/opt/python@3.12/bin/python3.12)"

if [ ! -x "$PYTHON312" ]; then
    echo "python3.12 not found. Install it first: brew install python@3.12 python-tk@3.12" >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR"
    # Homebrew's python@3.12 bottle links pyexpat against a libexpat symbol
    # newer than macOS may ship in /usr/lib/libexpat.1.dylib -- without this,
    # plistlib (which pip's SSL setup uses via platform.mac_ver()) fails and
    # even `python -m venv` can't bootstrap pip. Homebrew's own expat has the
    # symbol; the same env var is baked into the venv's activate script below
    # so every future `pip`/`python` call in it stays fixed, not just this one.
    if ! brew list --versions expat >/dev/null 2>&1; then
        brew install expat
    fi
    DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib" "$PYTHON312" -m venv "$VENV_DIR"

    cat >> "$VENV_DIR/bin/activate" <<'EOF'

# See install.sh -- fixes plistlib/platform.mac_ver() under this venv's
# Homebrew python@3.12 build on newer macOS versions.
_OLD_VIRTUAL_DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}"
DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH
EOF
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

echo "=== stage 1: core ==="
pip install "websockets==16.0" "numpy==2.5.2" "torch==2.6.0" "torchaudio==2.6.0" "soundfile==0.14.0" "silero-vad==6.2.1"

echo "=== stage 2: mlx ASR ==="
pip install "mlx==0.32.0" "parakeet_mlx==0.5.2"

echo "=== stage 3: Gemma 4 E2B (native MLX) ==="
pip install "mlx-vlm==0.6.13"

echo "=== stage 4: transformers stack (kokoro/chatterbox-tts need it transitively) ==="
pip install "transformers==5.15.0" "accelerate==1.14.0" "timm==1.0.28" "huggingface_hub==1.27.0" "torchvision==0.21.0"

echo "=== stage 5: kokoro ==="
pip install "kokoro==0.9.4"

echo "=== stage 6: chatterbox-tts ==="
pip install "chatterbox-tts==0.1.7"

echo "=== stage 7: sounddevice (TestChatUI.py only) ==="
pip install "sounddevice==0.5.5"

echo "=== stage 8: re-pin transformers ==="
# chatterbox-tts==0.1.7 declares transformers==5.2.0 as a hard pin and stage 6
# just downgraded to it -- but 5.2.0 predates gemma4 support (AutoConfig
# raises KeyError('gemma4') on it). 5.15.0 is confirmed to still import
# chatterbox-tts/kokoro fine despite the declared conflict pip warns about
# here; only pip's dependency metadata is stale, not an actual runtime break.
pip install "transformers==5.15.0"

echo
echo "Done. Activate with: source $VENV_DIR/bin/activate"
