"""
Enhanced audio captioning system using Qwen2-Audio and LLM refinement.
Uses VLLM for both models with flash attention and optimized batch processing.
"""

import os
import json
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AudioCaptionConfig:
    """Configuration for audio captioning pipeline."""
    # Audio model settings
    audio_model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    audio_max_tokens: int = 512
    audio_temperature: float = 0.7
    audio_top_p: float = 0.9
    
    # LLM refinement settings
    llm_model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_max_tokens: int = 256
    llm_temperature: float = 0.3
    llm_top_p: float = 0.95
    
    # VLLM settings
    tensor_parallel_size: int = 1  # Number of GPUs for tensor parallelism
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256  # Maximum sequences to process in parallel
    max_num_batched_tokens: int = 32768  # Maximum tokens in a batch
    enable_prefix_caching: bool = True
    
    # Processing settings
    audio_batch_size: int = 32
    llm_batch_size: int = 128
    max_audio_duration: float = 30.0  # seconds
    sample_rate: int = 16000  # Qwen2-Audio expects 16kHz


class AudioCaptionEnhancer:
    """Enhanced audio captioning with GPU-accelerated models."""
    
    def __init__(self, config: AudioCaptionConfig = None):
        self.config = config or AudioCaptionConfig()
        self.audio_model = None
        self.llm_model = None
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize VLLM models with optimized settings."""
        try:
            from vllm import LLM, SamplingParams
            from vllm.multimodal import MultiModalData
            
            # Initialize Qwen2-Audio with VLLM
            print("Initializing Qwen2-Audio with VLLM...")
            self.audio_model = LLM(
                model=self.config.audio_model_name,
                trust_remote_code=True,
                dtype="auto",  # Will use float16/bfloat16 automatically
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                enable_prefix_caching=self.config.enable_prefix_caching,
                max_model_len=2048,  # Adjust based on model requirements
            )
            
            # Initialize LLM for refinement
            print("Initializing Qwen2.5 LLM with VLLM...")
            self.llm_model = LLM(
                model=self.config.llm_model_name,
                trust_remote_code=True,
                dtype="auto",
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                enable_prefix_caching=self.config.enable_prefix_caching,
                max_model_len=4096,
            )
            
            # Create sampling params
            self.audio_sampling_params = SamplingParams(
                temperature=self.config.audio_temperature,
                top_p=self.config.audio_top_p,
                max_tokens=self.config.audio_max_tokens,
            )
            
            self.llm_sampling_params = SamplingParams(
                temperature=self.config.llm_temperature,
                top_p=self.config.llm_top_p,
                max_tokens=self.config.llm_max_tokens,
            )
            
        except ImportError:
            print("VLLM not found. Installing required packages...")
            os.system("pip install vllm flash-attn transformers accelerate")
            # Retry after installation
            self._initialize_models()
            
    def preprocess_audio(self, audio_path: Union[str, bytes, io.BytesIO]) -> Dict:
        """Preprocess audio for Qwen2-Audio model."""
        try:
            # Load audio
            if isinstance(audio_path, (bytes, io.BytesIO)):
                if isinstance(audio_path, bytes):
                    audio_path = io.BytesIO(audio_path)
                audio, sr = librosa.load(audio_path, sr=None, mono=True)
            else:
                audio, sr = librosa.load(str(audio_path), sr=None, mono=True)
            
            # Resample to 16kHz if needed (Qwen2-Audio requirement)
            if sr != self.config.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.config.sample_rate)
                sr = self.config.sample_rate
            
            # Limit duration
            max_samples = int(self.config.max_audio_duration * sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Convert to format expected by model
            audio_dict = {
                "array": audio,
                "sampling_rate": sr,
                "duration": len(audio) / sr
            }
            
            return audio_dict
            
        except Exception as e:
            raise Exception(f"Failed to preprocess audio: {e}")
    
    def generate_audio_captions_batch(self, audio_data_list: List[Dict]) -> List[str]:
        """Generate captions for a batch of audio files using Qwen2-Audio."""
        from vllm import SamplingParams
        
        # Prepare prompts for batch processing
        prompts = []
        for audio_data in audio_data_list:
            # Qwen2-Audio expects specific prompt format
            prompt = (
                "<|audio_bos|><|AUDIO|><|audio_eos|>"
                "Please describe this audio in detail. Include information about "
                "the sounds, their characteristics, duration, and any notable features."
            )
            prompts.append({
                "prompt": prompt,
                "multi_modal_data": {"audio": audio_data["array"]}
            })
        
        # Generate captions in batch
        outputs = self.audio_model.generate(
            prompts,
            sampling_params=self.audio_sampling_params,
            use_tqdm=False  # We'll handle progress externally
        )
        
        # Extract generated captions
        captions = [output.outputs[0].text.strip() for output in outputs]
        return captions
    
    def refine_captions_batch(self, caption_data_list: List[Dict]) -> List[str]:
        """Refine captions using LLM with original and generated captions."""
        prompts = []
        
        for data in caption_data_list:
            original_caption = data.get("original_caption", "")
            generated_caption = data.get("generated_caption", "")
            filename = data.get("filename", "audio")
            
            prompt = f"""You are an expert at creating high-quality audio captions. 
Given the following information about an audio file, create a refined, accurate, and descriptive caption.

Filename: {filename}
Original caption/description: {original_caption}
AI-generated caption: {generated_caption}

Create a refined caption that:
1. Combines the best information from both sources
2. Is accurate and descriptive
3. Removes any redundancy or contradictions
4. Is concise but informative (1-2 sentences)
5. Uses natural, fluent language

Refined caption:"""
            
            prompts.append(prompt)
        
        # Generate refined captions in batch
        outputs = self.llm_model.generate(
            prompts,
            sampling_params=self.llm_sampling_params,
            use_tqdm=False
        )
        
        # Extract refined captions
        refined_captions = [output.outputs[0].text.strip() for output in outputs]
        return refined_captions
    
    def process_audio_files(self, 
                          audio_files: List[Tuple[Union[str, bytes], str, Dict]],
                          show_progress: bool = True) -> List[Dict]:
        """
        Process audio files to generate enhanced captions.
        
        Args:
            audio_files: List of (audio_path_or_bytes, original_caption, metadata)
            show_progress: Whether to show progress bar
            
        Returns:
            List of dicts with enhanced caption data
        """
        results = []
        
        # Process in batches for efficiency
        audio_batch_size = self.config.audio_batch_size
        llm_batch_size = self.config.llm_batch_size
        
        # Phase 1: Generate audio captions
        print("Phase 1: Generating audio captions with Qwen2-Audio...")
        audio_captions = []
        
        for i in tqdm(range(0, len(audio_files), audio_batch_size), 
                     disable=not show_progress,
                     desc="Processing audio batches"):
            batch = audio_files[i:i + audio_batch_size]
            audio_data_batch = []
            
            # Preprocess audio files in batch
            for audio_source, _, _ in batch:
                try:
                    audio_data = self.preprocess_audio(audio_source)
                    audio_data_batch.append(audio_data)
                except Exception as e:
                    print(f"Failed to preprocess audio: {e}")
                    audio_data_batch.append(None)
            
            # Filter out failed preprocessing
            valid_indices = [j for j, data in enumerate(audio_data_batch) if data is not None]
            valid_audio_data = [audio_data_batch[j] for j in valid_indices]
            
            if valid_audio_data:
                # Generate captions for valid audio
                captions = self.generate_audio_captions_batch(valid_audio_data)
                
                # Map back to original indices
                caption_results = [""] * len(batch)
                for idx, caption in zip(valid_indices, captions):
                    caption_results[idx] = caption
                
                audio_captions.extend(caption_results)
            else:
                audio_captions.extend([""] * len(batch))
        
        # Phase 2: Refine captions with LLM
        print("\nPhase 2: Refining captions with Qwen2.5 LLM...")
        
        # Prepare data for LLM refinement
        caption_data = []
        for i, (audio_source, original_caption, metadata) in enumerate(audio_files):
            filename = metadata.get("original_filename", f"audio_{i}")
            caption_data.append({
                "original_caption": original_caption,
                "generated_caption": audio_captions[i],
                "filename": filename,
                "metadata": metadata
            })
        
        # Process in LLM batches
        refined_captions = []
        for i in tqdm(range(0, len(caption_data), llm_batch_size),
                     disable=not show_progress,
                     desc="Refining captions"):
            batch = caption_data[i:i + llm_batch_size]
            batch_refined = self.refine_captions_batch(batch)
            refined_captions.extend(batch_refined)
        
        # Compile results
        for i, (audio_source, original_caption, metadata) in enumerate(audio_files):
            result = {
                "original_caption": original_caption,
                "generated_caption": audio_captions[i],
                "refined_caption": refined_captions[i],
                "metadata": metadata
            }
            results.append(result)
        
        return results
    
    def process_dataset_with_enhancement(self,
                                       matched_samples: List[Tuple],
                                       output_path: str = None) -> List[Dict]:
        """
        Process entire dataset with caption enhancement.
        
        Args:
            matched_samples: List of (audio_path, caption, metadata) tuples
            output_path: Optional path to save results
            
        Returns:
            Enhanced results
        """
        print(f"Processing {len(matched_samples)} audio files with caption enhancement...")
        
        # Process all files
        results = self.process_audio_files(matched_samples)
        
        # Save results if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"Saved enhanced captions to {output_path}")
        
        # Print statistics
        total = len(results)
        with_generated = sum(1 for r in results if r["generated_caption"])
        with_refined = sum(1 for r in results if r["refined_caption"])
        
        print(f"\nProcessing complete:")
        print(f"  Total files: {total}")
        print(f"  Successfully generated captions: {with_generated}")
        print(f"  Successfully refined captions: {with_refined}")
        
        return results


# Integration with existing DatasetProcessor
class EnhancedAudioProcessor:
    """Extended AudioProcessor with caption enhancement capabilities."""
    
    def __init__(self, 
                 target_sr: int = 48000,
                 output_format: str = "flac",
                 enable_caption_enhancement: bool = True,
                 caption_config: AudioCaptionConfig = None):
        self.target_sr = target_sr
        self.output_format = output_format
        self.enable_caption_enhancement = enable_caption_enhancement
        
        if enable_caption_enhancement:
            self.caption_enhancer = AudioCaptionEnhancer(caption_config)
        else:
            self.caption_enhancer = None
    
    def process_audio_file_with_caption(self, 
                                      audio_path: Union[str, bytes, io.BytesIO],
                                      original_caption: str = "",
                                      metadata: Dict = None) -> Tuple[bytes, Dict]:
        """Process audio and optionally enhance caption."""
        # First, process audio as usual
        try:
            # Load and process audio
            if isinstance(audio_path, (bytes, io.BytesIO)):
                if isinstance(audio_path, bytes):
                    audio_path = io.BytesIO(audio_path)
                audio, sr = librosa.load(audio_path, sr=None, mono=False)
            else:
                audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
            
            # Resample if needed
            if sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
                sr = self.target_sr
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Save to bytes
            output_buffer = io.BytesIO()
            sf.write(output_buffer, audio, sr, format=self.output_format.upper())
            output_buffer.seek(0)
            audio_bytes = output_buffer.read()
            
            # Basic metadata
            result_metadata = {
                "sample_rate": sr,
                "duration": len(audio) / sr,
                "channels": 1,
                "format": self.output_format
            }
            
            # Enhance caption if enabled
            if self.enable_caption_enhancement and self.caption_enhancer:
                enhancement_results = self.caption_enhancer.process_audio_files(
                    [(audio_path, original_caption, metadata or {})],
                    show_progress=False
                )[0]
                
                result_metadata.update({
                    "original_caption": original_caption,
                    "generated_caption": enhancement_results["generated_caption"],
                    "refined_caption": enhancement_results["refined_caption"]
                })
            
            if metadata:
                result_metadata.update(metadata)
            
            return audio_bytes, result_metadata
            
        except Exception as e:
            raise Exception(f"Failed to process audio with caption: {e}")


if __name__ == "__main__":
    # Example usage
    config = AudioCaptionConfig(
        tensor_parallel_size=1,  # Adjust based on available GPUs
        gpu_memory_utilization=0.9,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
        audio_batch_size=32,
        llm_batch_size=128
    )
    
    enhancer = AudioCaptionEnhancer(config)
    
    # Example processing
    test_files = [
        ("/path/to/audio1.wav", "Original caption 1", {"filename": "audio1.wav"}),
        ("/path/to/audio2.mp3", "Original caption 2", {"filename": "audio2.mp3"}),
    ]
    
    results = enhancer.process_audio_files(test_files)
    
    for result in results:
        print(f"\nFile: {result['metadata'].get('filename', 'Unknown')}")
        print(f"Original: {result['original_caption']}")
        print(f"Generated: {result['generated_caption']}")
        print(f"Refined: {result['refined_caption']}")