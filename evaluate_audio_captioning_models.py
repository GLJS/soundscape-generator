#!/usr/bin/env python3
"""
Evaluate multiple audio captioning models on a single audio example from AudioDataFull dataset.

Models to evaluate:
1. Qwen2.5-Omni
2. Audio Flamingo 3
3. Phi-4 Multimodal
4. Mistral Voxtral
"""

# from dotenv import load_dotenv
# load_dotenv()
import torch
import pandas as pd
import soundfile as sf
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
import ast

# Suppress warnings
warnings.filterwarnings('ignore')


class AudioCaptioningEvaluator:
    """Evaluates multiple audio captioning models on a single audio sample."""
    
    def __init__(self, audio_dir: str = "/scratch-shared/gwijngaard/laion/AudioDataFull"):
        self.audio_dir = Path(audio_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def load_single_example(self) -> Tuple[Path, str, Dict[str, Any]]:
        """Load a single example from AudioDataFull dataset."""
        print("\n=== Loading Audio Example ===")
        
        # Load metadata CSV
        csv_path = self.audio_dir / "df_lemm_srl_path.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
            
        # Load first valid audio example
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} entries from metadata")
        
        for idx, row in df.iterrows():
            # Convert path
            local_path = row['path'].replace(
                '/root/share/AudioData/', 
                str(self.audio_dir) + '/'
            )
            audio_path = Path(local_path)
            
            if audio_path.exists():
                # Extract caption
                caption = row.get('description', '')
                if isinstance(caption, str) and caption.startswith('['):
                    try:
                        desc_list = ast.literal_eval(caption)
                        if desc_list and isinstance(desc_list[0], str):
                            caption = desc_list[0].replace('comment=', '').split('\\;')[0]
                    except (ValueError, SyntaxError):
                        pass
                
                # If no good caption, use filename
                if not caption or caption == '[]':
                    caption = row['filename'].replace('.wav', '').replace('.mp3', '').replace('.flac', '')
                
                metadata = {
                    'filename': row['filename'],
                    'dataset_root': row.get('dataset_root', ''),
                    'original_path': row.get('path', ''),
                    'index': idx
                }
                
                print(f"Found audio file: {audio_path.name}")
                print(f"Original description: {caption}")
                return audio_path, caption, metadata
                
        raise FileNotFoundError("No valid audio files found in dataset")
        
    def evaluate_qwen25_omni(self, audio_path: Path) -> Optional[str]:
        """Evaluate Qwen2.5-Omni model."""
        print("\n=== Evaluating Qwen2.5-Omni ===")
        
        try:
            # For now, return a placeholder as Qwen2.5-Omni has CUDA issues
            print("Qwen2.5-Omni encountered CUDA errors - skipping for now")
            print("The model requires specific CUDA configurations")
            return "Qwen2.5-Omni - Skipped due to CUDA compatibility issues"
            
        except Exception as e:
            print(f"Error with Qwen2.5-Omni: {e}")
            return None
            
    def evaluate_audio_flamingo3(self, audio_path: Path) -> Optional[str]:
        """Evaluate Audio Flamingo 3 model."""
        print("\n=== Evaluating Audio Flamingo 3 ===")
        
        try:
            # Note: Audio Flamingo 3 requires custom implementation
            # This is a placeholder for the actual implementation
            print("Audio Flamingo 3 requires custom loader - implementation pending")
            print("Model available at: nvidia/audio-flamingo-3")
            
            # Placeholder implementation
            # In practice, you would need to:
            # 1. Clone the GitHub repo
            # 2. Use their custom loading mechanism
            # 3. Or use their demo API if available
            
            return "Audio Flamingo 3 - Not implemented (requires custom loader)"
            
        except Exception as e:
            print(f"Error with Audio Flamingo 3: {e}")
            return None
            
    def evaluate_phi4_multimodal(self, audio_path: Path) -> Optional[str]:
        """Evaluate Phi-4 Multimodal model."""
        print("\n=== Evaluating Phi-4 Multimodal ===")
        
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            model_name = "microsoft/Phi-4-multimodal-instruct"
            print(f"Loading {model_name}...")
            
            # Load processor and model
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
                attn_implementation="eager"  # Disable Flash Attention
            )
            
            # Load speech adapter
            try:
                model.load_adapter(model_name, adapter_name="speech", device_map="auto", 
                                 adapter_kwargs={"subfolder": 'speech-lora'})
                print("Loaded speech adapter")
            except Exception:
                print("Warning: Could not load speech adapter")
            
            # Read audio
            audio_data, sample_rate = sf.read(str(audio_path))
            
            # Prepare input
            prompt = "<|audio_1|> Describe this audio in detail."
            inputs = processor(text=prompt, audio=audio_data, sampling_rate=sample_rate, 
                             return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=128)
            
            caption = processor.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated caption: {caption}")
            
            # Clean up
            del model, processor
            torch.cuda.empty_cache()
            
            return caption
            
        except Exception as e:
            print(f"Error with Phi-4 Multimodal: {e}")
            return None
            
    def evaluate_mistral_voxtral(self, audio_path: Path) -> Optional[str]:
        """Evaluate Mistral Voxtral model."""
        print("\n=== Evaluating Mistral Voxtral ===")
        
        try:
            # Voxtral requires vLLM with audio support
            print("Mistral Voxtral requires vLLM with audio support")
            print("To use: vllm serve mistralai/Voxtral-Mini-3B-2507 --tokenizer_mode mistral")
            
            # Alternative: try direct loading if transformers support is added
            from transformers import AutoModelForCausalLM, AutoProcessor
            
            model_name = "mistralai/Voxtral-Mini-3B-2507"
            print(f"Attempting to load {model_name} with transformers...")
            
            # This might not work without vLLM, but worth trying
            try:
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                # Clean up
                del model, processor
                torch.cuda.empty_cache()
            except Exception:
                pass
            
            # Process audio - implementation depends on Voxtral's specific requirements
            return "Voxtral - Requires vLLM server (not implemented in direct mode)"
            
        except Exception as e:
            print(f"Error with Mistral Voxtral: {e}")
            print("Note: Voxtral is designed to run with vLLM server")
            return None
            
    def evaluate_whisper(self, audio_path: Path) -> Optional[str]:
        """Evaluate Whisper model as a baseline."""
        print("\n=== Evaluating Whisper (Baseline) ===")
        
        try:
            from transformers import pipeline
            
            # Initialize Whisper pipeline
            print("Loading openai/whisper-base...")
            transcriber = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-base",
                device=self.device
            )
            
            # Transcribe audio
            print("Transcribing audio...")
            result = transcriber(
                str(audio_path),
                generate_kwargs={"task": "transcribe", "language": "en"}
            )
            
            caption = result['text'].strip()
            if not caption or caption == "." * len(caption):
                # If no speech detected, describe as environmental sound
                caption = "Environmental sound or non-speech audio detected"
            
            print(f"Generated caption: {caption}")
            
            # Clean up
            del transcriber
            torch.cuda.empty_cache()
            
            return caption
            
        except Exception as e:
            print(f"Error with Whisper: {e}")
            return None
            
    def run_evaluation(self):
        """Run evaluation on all models."""
        # Load example
        audio_path, original_caption, metadata = self.load_single_example()
        
        print(f"\n{'='*60}")
        print(f"Audio File: {metadata['filename']}")
        print(f"Original Description: {original_caption}")
        print(f"{'='*60}")
        
        # Results dictionary
        results = {
            "audio_file": metadata['filename'],
            "original_description": original_caption,
            "model_captions": {}
        }
        
        # Evaluate each model
        # Qwen2.5-Omni
        caption = self.evaluate_qwen25_omni(audio_path)
        if caption:
            results["model_captions"]["Qwen2.5-Omni"] = caption
            
        # Audio Flamingo 3
        caption = self.evaluate_audio_flamingo3(audio_path)
        if caption:
            results["model_captions"]["Audio Flamingo 3"] = caption
            
        # Phi-4 Multimodal
        caption = self.evaluate_phi4_multimodal(audio_path)
        if caption:
            results["model_captions"]["Phi-4 Multimodal"] = caption
            
        # Mistral Voxtral
        caption = self.evaluate_mistral_voxtral(audio_path)
        if caption:
            results["model_captions"]["Mistral Voxtral"] = caption
            
        # Whisper (Baseline)
        caption = self.evaluate_whisper(audio_path)
        if caption:
            results["model_captions"]["Whisper (Baseline)"] = caption
            
        # Print summary
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Audio: {results['audio_file']}")
        print(f"Original: {results['original_description']}")
        print(f"\nModel Captions:")
        for model, caption in results['model_captions'].items():
            print(f"\n{model}:")
            print(f"  {caption}")
            
        return results


def main():
    """Main function."""
    evaluator = AudioCaptioningEvaluator()
    results = evaluator.run_evaluation()
    
    # Save results
    import json
    output_path = "audio_captioning_evaluation_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()