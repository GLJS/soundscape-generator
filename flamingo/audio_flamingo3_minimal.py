#!/usr/bin/env python3
"""
Minimal AudioFlamingo3 example for audio captioning.
Loads the model and generates a caption for a single audio file.
"""
from dotenv import load_dotenv

load_dotenv()
import os
import sys
import torch
from huggingface_hub import snapshot_download

# Add the audio-flamingo directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'audio-flamingo'))

import llava


def caption_audio(audio_file_path, prompt_text=None):
    """
    Generate a caption for an audio file using AudioFlamingo3.
    
    Args:
        audio_file_path: Path to the audio file
        prompt_text: Optional custom prompt (defaults to general description request)
    
    Returns:
        Generated caption string
    """
    # Default prompt if none provided
    if prompt_text is None:
        prompt_text = "Please describe this audio in detail. Include information about the sounds, their characteristics, duration, and any notable features."
    
    # Download model from Hugging Face
    print("Loading AudioFlamingo3 model...")
    MODEL_BASE = snapshot_download(repo_id="nvidia/audio-flamingo-3")
    
    # Load the model
    model = llava.load(MODEL_BASE, model_base=None)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get default generation config
    generation_config = model.default_generation_config
    
    # Create Sound object from audio file
    sound = llava.Sound(audio_file_path)
    
    # Format prompt according to model requirements
    full_prompt = f"<sound>\n{prompt_text}"
    
    # Generate caption
    print(f"\nProcessing audio file: {audio_file_path}")
    print("Generating caption...")
    
    response = model.generate_content([sound, full_prompt], generation_config=generation_config)
    
    return response


def main():
    # Audio file in current directory
    audio_file = "AirplaneJet_DIGIMEGADISC-52.wav"
    
    # Check if file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found in current directory.")
        sys.exit(1)
    
    # Generate caption
    caption = caption_audio(audio_file)
    
    # Print result
    print("\n" + "="*50)
    print("GENERATED CAPTION:")
    print("="*50)
    print(caption)
    print("="*50)
    
    # Example with custom prompt
    print("\n\nTrying with a custom prompt...")
    custom_prompt = "What type of vehicle or machine can you hear in this audio? Describe the sound characteristics."
    custom_caption = caption_audio(audio_file, custom_prompt)
    
    print("\n" + "="*50)
    print("CUSTOM PROMPT CAPTION:")
    print("="*50)
    print(custom_caption)
    print("="*50)


if __name__ == "__main__":
    main()
