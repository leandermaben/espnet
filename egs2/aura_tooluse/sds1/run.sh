#!/usr/bin/env bash
# Aura Tool Use Agent Demo

set -e
set -u
set -o pipefail

# Check for HuggingFace token if using gated models
if [ -z "${HF_TOKEN:-}" ]; then
    echo "Warning: HF_TOKEN not set. Some models may not be accessible."
    echo "Set it with: export HF_TOKEN=your_token_here"
fi

# Default models
asr_model="pyf98/owsm_ctc_v3.1_1B"
llm_model="meta-llama/Llama-3.2-1B-Instruct"
tts_model="kan-bayashi/ljspeech_vits"
device="cuda"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --asr_model)
            asr_model="$2"
            shift 2
            ;;
        --llm_model)
            llm_model="$2"
            shift 2
            ;;
        --tts_model)
            tts_model="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting Aura Tool Use Agent Demo..."
echo "ASR Model: ${asr_model}"
echo "LLM Model: ${llm_model}"
echo "TTS Model: ${tts_model}"
echo "Device: ${device}"
echo ""

# Run the app
python app.py \
    --asr_model "${asr_model}" \
    --llm_model "${llm_model}" \
    --tts_model "${tts_model}" \
    --device "${device}" \
    --hf_token "${HF_TOKEN:-}"
