"""
Aura Tool Use Agent Demo for ESPnet-SDS.

This demo showcases a speech-to-speech conversational agent with tool use capabilities.
The agent can perform web searches and calculations while maintaining natural conversation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr
import numpy as np
import torch

# Add local path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pyscripts.utils.aura_agent.agent import AuraAgent
from pyscripts.utils.aura_agent.state import State

# Try to import ESPnet components
try:
    from espnet2.bin.asr_inference import Speech2Text
    from espnet2.bin.tts_inference import Text2Speech

    ESPNET_AVAILABLE = True
except ImportError:
    ESPNET_AVAILABLE = False
    print("Warning: ESPnet not available, using mock implementations")


# Global variables
state = None
agent = None
asr_model = None
tts_model = None
llm_model = None

latency_ASR = 0.0
latency_LM = 0.0
latency_TTS = 0.0


class SimpleLLMClient:
    """Simple LLM client wrapper for HuggingFace models."""

    def __init__(self, model_name: str, access_token: Optional[str] = None):
        """Initialize LLM client.

        Args:
            model_name: HuggingFace model name.
            access_token: Optional HF access token.
        """
        self.model_name = model_name
        self.access_token = access_token
        self.pipeline = None
        self._load_model()

    def _load_model(self):
        """Load the LLM model."""
        try:
            from transformers import pipeline

            self.pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                token=self.access_token,
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
            )
            print(f"Loaded LLM: {self.model_name}")
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            self.pipeline = None

    def generate(self, messages: list) -> str:
        """Generate a response from messages.

        Args:
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            str: Generated response.
        """
        if self.pipeline is None:
            return "Error: LLM not available"

        try:
            # Format messages into a prompt
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n\n"
                elif role == "user":
                    prompt += f"User: {content}\n\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n\n"

            prompt += "Assistant: "

            # Generate response
            outputs = self.pipeline(
                prompt, max_new_tokens=256, return_full_text=False
            )
            response = outputs[0]["generated_text"].strip()

            return response

        except Exception as e:
            return f"Error generating response: {str(e)}"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Aura Tool Use Agent demo.")
    parser.add_argument(
        "--asr_model",
        type=str,
        default="pyf98/owsm_ctc_v3.1_1B",
        help="ASR model name or path",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="LLM model name",
    )
    parser.add_argument(
        "--tts_model",
        type=str,
        default="kan-bayashi/ljspeech_vits",
        help="TTS model name or path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for models",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace access token",
    )
    return parser.parse_args()


def load_models(args):
    """Load ASR, LLM, and TTS models.

    Args:
        args: Command line arguments.
    """
    global asr_model, llm_model, tts_model, agent, state

    print("Loading models...")

    # Load ASR
    if ESPNET_AVAILABLE:
        try:
            asr_model = Speech2Text.from_pretrained(
                model_tag=args.asr_model,
                device=args.device,
            )
            print(f"Loaded ASR model: {args.asr_model}")
        except Exception as e:
            print(f"Failed to load ASR model: {e}")
            asr_model = None
    else:
        asr_model = None

    # Load LLM
    llm_model = SimpleLLMClient(args.llm_model, args.hf_token)

    # Load TTS
    if ESPNET_AVAILABLE:
        try:
            tts_model = Text2Speech.from_pretrained(
                model_tag=args.tts_model,
                device=args.device,
            )
            print(f"Loaded TTS model: {args.tts_model}")
        except Exception as e:
            print(f"Failed to load TTS model: {e}")
            tts_model = None
    else:
        tts_model = None

    # Initialize agent
    agent = AuraAgent(llm_model)
    state = State()

    print("All models loaded successfully!")


def transcribe_audio(audio_data: np.ndarray, sample_rate: int) -> str:
    """Transcribe audio using ASR model.

    Args:
        audio_data: Audio waveform.
        sample_rate: Audio sample rate.

    Returns:
        str: Transcribed text.
    """
    global latency_ASR

    if asr_model is None:
        return "[ASR not available - please speak your message]"

    try:
        start = time.time()

        # Resample if needed
        if sample_rate != 16000:
            import librosa

            audio_data = librosa.resample(
                audio_data, orig_sr=sample_rate, target_sr=16000
            )

        # Run ASR
        nbests = asr_model(audio_data)
        text = nbests[0].text if nbests else ""

        latency_ASR = time.time() - start
        return text

    except Exception as e:
        print(f"ASR error: {e}")
        return f"[ASR error: {str(e)}]"


def synthesize_speech(text: str) -> Optional[Tuple[int, np.ndarray]]:
    """Synthesize speech from text using TTS model.

    Args:
        text: Text to synthesize.

    Returns:
        Tuple of (sample_rate, audio_array) or None.
    """
    global latency_TTS

    if tts_model is None:
        return None

    try:
        start = time.time()

        # Generate speech
        wav = tts_model(text)["wav"]
        audio_array = wav.cpu().numpy()

        # Get sample rate from model config
        sample_rate = getattr(tts_model, "fs", 22050)

        latency_TTS = time.time() - start
        return (sample_rate, audio_array)

    except Exception as e:
        print(f"TTS error: {e}")
        return None


def process_audio(audio: Optional[Tuple[int, np.ndarray]]) -> Tuple:
    """Process audio input through the full pipeline.

    Args:
        audio: Tuple of (sample_rate, audio_data) or None.

    Returns:
        Tuple of (transcript, response_text, response_audio, latency_info, history_html).
    """
    global state, agent, latency_ASR, latency_LM, latency_TTS

    if audio is None:
        return "", "", None, "", format_history()

    sample_rate, audio_data = audio

    # Step 1: Transcribe audio
    transcript = transcribe_audio(audio_data, sample_rate)

    if not transcript or transcript.startswith("["):
        return transcript, "", None, "", format_history()

    # Step 2: Run agent
    start = time.time()
    response_text = agent.run(transcript, state)
    latency_LM = time.time() - start

    # Step 3: Synthesize speech
    response_audio = synthesize_speech(response_text)

    # Format latency info
    latency_info = (
        f"ASR: {latency_ASR:.2f}s | "
        f"LLM: {latency_LM:.2f}s | "
        f"TTS: {latency_TTS:.2f}s | "
        f"Total: {latency_ASR + latency_LM + latency_TTS:.2f}s"
    )

    # Format conversation history
    history_html = format_history()

    return transcript, response_text, response_audio, latency_info, history_html


def format_history() -> str:
    """Format conversation history as HTML.

    Returns:
        str: HTML formatted conversation history.
    """
    if state is None or not state.messages:
        return "<p>No conversation yet. Start by speaking!</p>"

    html = '<div style="max-height: 400px; overflow-y: auto;">'

    for msg in state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            html += (
                f'<div style="margin: 10px; padding: 10px; '
                f'background-color: #e3f2fd; border-radius: 10px;">'
                f'<strong>👤 You:</strong> {content}</div>'
            )
        else:
            html += (
                f'<div style="margin: 10px; padding: 10px; '
                f'background-color: #f1f8e9; border-radius: 10px;">'
                f'<strong>🤖 Aura:</strong> {content}</div>'
            )

    # Add observations if any
    if state.observations:
        html += '<div style="margin: 10px; padding: 10px; background-color: #fff3e0; border-radius: 10px;">'
        html += '<strong>🔧 Tool Usage:</strong><ul>'
        for obs in state.observations[-3:]:
            html += f'<li>{obs["action"]} → {obs["observation"][:100]}...</li>'
        html += "</ul></div>"

    html += "</div>"
    return html


def clear_conversation():
    """Clear conversation history."""
    global state
    if state:
        state.clear()
    return "", "", None, "", format_history()


def create_demo():
    """Create Gradio demo interface."""
    with gr.Blocks(title="Aura Tool Use Agent") as demo:
        gr.Markdown(
            """
            # 🎙️ Aura Tool Use Agent Demo

            This demo showcases a speech-to-speech conversational agent with tool use capabilities.
            The agent can:
            - Engage in natural conversation
            - Search the web for information
            - Perform calculations

            **How to use:**
            1. Click the microphone button and speak your question
            2. The agent will transcribe your speech, process it, and respond
            3. View the conversation history and tool usage below

            **Example queries:**
            - "What is the capital of France?" (web search)
            - "Calculate 25 times 17" (calculator)
            - "Hello, how are you?" (chat)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎤 Speak your message",
                )
                clear_btn = gr.Button("🗑️ Clear Conversation")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="🔊 Agent Response", autoplay=True)

        with gr.Row():
            transcript_output = gr.Textbox(label="📝 Your message (ASR output)")
            response_output = gr.Textbox(label="💬 Agent response (text)")

        latency_output = gr.Textbox(label="⏱️ Latency Information")
        history_output = gr.HTML(label="💭 Conversation History")

        # Event handlers
        audio_input.stop_recording(
            fn=process_audio,
            inputs=[audio_input],
            outputs=[
                transcript_output,
                response_output,
                audio_output,
                latency_output,
                history_output,
            ],
        )

        clear_btn.click(
            fn=clear_conversation,
            inputs=[],
            outputs=[
                transcript_output,
                response_output,
                audio_output,
                latency_output,
                history_output,
            ],
        )

        # Load initial history
        demo.load(fn=format_history, inputs=[], outputs=[history_output])

    return demo


if __name__ == "__main__":
    args = parse_args()

    # Load models
    load_models(args)

    # Create and launch demo
    demo = create_demo()
    demo.launch(share=True, server_name="0.0.0.0")
