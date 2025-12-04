#!/usr/bin/env bash
# Setup script for Aura Tool Use Agent

set -e
set -u
set -o pipefail

echo "Setting up Aura Tool Use Agent environment..."

# Install required Python packages
pip install gradio>=4.0.0
pip install transformers>=4.30.0
pip install requests
pip install librosa

echo ""
echo "Setup complete!"
echo ""
echo "To run the demo:"
echo "  1. Set your HuggingFace token (optional, for gated models):"
echo "     export HF_TOKEN=your_token_here"
echo ""
echo "  2. Run the demo:"
echo "     bash run.sh"
echo ""
echo "The demo will be available at http://localhost:7860"
