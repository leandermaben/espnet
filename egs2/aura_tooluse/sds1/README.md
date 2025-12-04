# Aura Tool Use Agent Demo

This demo showcases an end-to-end spoken dialogue system with **tool use capabilities** based on the [Aura](https://github.com/Sentientia/Aura) framework. The agent can engage in natural conversation while intelligently using tools like web search and calculators when needed.

## Overview

The Aura Tool Use Agent combines:
- **ASR (Automatic Speech Recognition)**: Transcribes user speech to text
- **Tool-Augmented LLM**: Reasons about when and how to use tools using ReAct-style prompting
- **TTS (Text-to-Speech)**: Converts agent responses back to speech

### Key Features

1. **Natural Conversation**: Engages in fluid dialogue without tools when appropriate
2. **Web Search**: Retrieves current information from the internet when needed
3. **Calculator**: Performs mathematical computations
4. **ReAct Reasoning**: Uses a Thought-Action-Input pattern for transparent decision-making

## Quick Start

### Prerequisites

1. Install ESPnet (if not already installed):
```bash
git clone https://github.com/espnet/espnet
cd espnet/tools
./setup_anaconda.sh
make
```

2. Install additional dependencies:
```bash
cd egs2/aura_tooluse/sds1
bash setup.sh
```

### Running the Demo

1. (Optional) Set your HuggingFace token for gated models:
```bash
export HF_TOKEN=your_token_here
```

2. Run the demo:
```bash
bash run.sh
```

3. Open the provided URL (usually http://localhost:7860) in your browser

4. Click the microphone button and start speaking!

## Usage Examples

### General Conversation
**User**: "Hello! How are you today?"
**Agent**: Responds directly without using tools

### Web Search
**User**: "What is the capital of France?"
**Agent**:
- Thought: I need current factual information
- Action: web_search
- Input: capital of France
- Response: Provides answer based on search results

### Calculator
**User**: "What's 127 times 83?"
**Agent**:
- Thought: This requires calculation
- Action: calculator
- Input: 127 * 83
- Response: Provides the computed result

## Architecture

### Agent Components

The demo follows ESPnet's SDS1 recipe structure with these key components:

```
egs2/aura_tooluse/sds1/
├── app.py                          # Main Gradio application
├── run.sh                          # Launch script
├── setup.sh                        # Environment setup
└── pyscripts/utils/aura_agent/    # Agent implementation
    ├── agent.py                    # ReAct agent with tool use
    ├── action.py                   # Tool action classes
    └── state.py                    # Conversation state management
```

### Tool Use Flow

1. **User Input**: Audio is captured via microphone
2. **ASR**: Speech is transcribed to text
3. **Agent Processing**:
   - LLM generates Thought-Action-Input response
   - If action is a tool (web_search, calculator), execute it
   - If action is chat, return response directly
4. **TTS**: Text response is synthesized to speech
5. **Output**: Audio is played to user

### ReAct Prompting

The agent uses a ReAct (Reasoning + Acting) pattern:

```
Thought: [reasoning about what to do]
Action: [tool_name or chat]
Input: [input for the tool]
```

This makes the agent's decision-making transparent and allows it to chain multiple tool uses if needed.

## Customization

### Using Different Models

You can specify custom models when running:

```bash
bash run.sh \
  --asr_model "espnet/owsm_ctc_v3.2_ft_1B" \
  --llm_model "meta-llama/Llama-3.2-3B-Instruct" \
  --tts_model "kan-bayashi/vctk_multi_spk_vits" \
  --device "cuda"
```

### Adding New Tools

To add a new tool, create a new Action class in `pyscripts/utils/aura_agent/action.py`:

```python
class MyToolAction(Action):
    def execute(self, state):
        # Your tool implementation
        result = do_something(self.payload)
        state.add_observation(f"Tool: {self.payload}", result)
        return result
```

Then update the agent's `parse_action` method and system prompt to include the new tool.

## Technical Details

### Models Used

- **ASR**: OWSM CTC v3.1 (1B parameters) - Multilingual ASR
- **LLM**: Llama 3.2 (1B parameters) - Instruction-tuned for dialogue
- **TTS**: VITS (LJSpeech) - Natural speech synthesis

### Performance

Typical latencies (on GPU):
- ASR: ~0.2-0.5s
- LLM: ~1-3s (depending on model size and tool use)
- TTS: ~0.3-0.7s
- **Total**: ~2-4s per turn

### Memory Requirements

- With 1B LLM: ~4GB GPU memory
- With 3B LLM: ~8GB GPU memory
- CPU mode: Works but slower (~10-30s per turn)

## Differences from Original Aura

This demo is a simplified version focused on the core tool use functionality:

**Included**:
- ✅ ReAct-style tool use agent
- ✅ Web search capability
- ✅ Calculator tool
- ✅ Speech-to-speech interface
- ✅ ESPnet ASR/TTS integration

**Not Included** (following feedback on PR #6100):
- ❌ Evaluation framework
- ❌ Dialog State Tracking (DST)
- ❌ Accent-adaptive ASR
- ❌ Calendar/Email integration (requires external API setup)
- ❌ Contact management

These were omitted to create a cleaner, more focused example that follows ESPnet's recipe conventions without duplicating existing functionality.

## Troubleshooting

### Models Not Loading
- Ensure you have sufficient GPU memory (4GB+ recommended)
- Try using CPU mode: `--device cpu` (slower but works)
- Check your HF_TOKEN if using gated models like Llama

### Audio Not Working
- Use Chrome browser (recommended for WebRTC support)
- Grant microphone permissions when prompted
- Check that your microphone is working in system settings

### Import Errors
- Run `setup.sh` to install all dependencies
- Verify ESPnet is properly installed: `python -c "import espnet"`

## Citation

If you use this demo in your research, please cite both ESPnet and Aura:

```bibtex
@inproceedings{watanabe2018espnet,
  title={{ESPnet}: End-to-end speech processing toolkit},
  author={Watanabe, Shinji and Hori, Takaaki and others},
  booktitle={Interspeech},
  pages={2207--2211},
  year={2018}
}

@article{aura2024,
  title={Aura: Natural Speech Interface with Tool Use},
  author={Sentientia Team},
  url={https://github.com/Sentientia/Aura},
  year={2024}
}
```

## References

- [ESPnet](https://github.com/espnet/espnet)
- [Aura](https://github.com/Sentientia/Aura)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [ESPnet-SDS Toolkit](https://github.com/espnet/espnet/tree/master/egs2/TEMPLATE/sds1)

## License

This demo follows ESPnet's Apache 2.0 license. The Aura framework is used as inspiration for the agent architecture.
