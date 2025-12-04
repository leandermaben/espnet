#!/usr/bin/env bash

# Path configuration for Aura Tool Use Agent

MAIN_ROOT=$PWD/../../..
KALDI_ROOT=$MAIN_ROOT/tools/kaldi

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present, can not continue!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$MAIN_ROOT/tools/chainer_ctc/ext/warp-ctc/build
. "${MAIN_ROOT}"/tools/activate_python.sh
[ -f "${MAIN_ROOT}"/tools/extra_path.sh ] && . "${MAIN_ROOT}"/tools/extra_path.sh
export PATH=$MAIN_ROOT/utils:$MAIN_ROOT/espnet/bin:$PATH

export OMP_NUM_THREADS=1

# Check installation
if ! python3 -c "import espnet" &> /dev/null; then
    echo "ESPnet is not installed. Please install ESPnet first."
    echo "See: https://github.com/espnet/espnet#installation"
    return 1
fi

# Check required packages
python3 -c "import gradio" &> /dev/null || echo "Warning: gradio not installed. Install with: pip install gradio"
python3 -c "import transformers" &> /dev/null || echo "Warning: transformers not installed. Install with: pip install transformers"
python3 -c "import requests" &> /dev/null || echo "Warning: requests not installed. Install with: pip install requests"
