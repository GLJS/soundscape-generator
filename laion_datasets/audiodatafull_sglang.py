#!/usr/bin/env python3
"""
AudioDataFull dataset processor using SGLang for caption generation.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor
from typing import List, Tuple, Dict
import argparse
from dotenv import load_dotenv
from utils_hybrid_streaming_sglang import HybridStreamingProcessor
from tqdm import tqdm
import json

load_dotenv()


class AudioDataFullProcessor(DatasetProcessor):
    """Processor for AudioDataFull dataset with SGLang captioning."""
    
    def __init__(self, audio_dir: str, metadata_path: str, output_dir: str, 
                 batch_size: int = 128, num_workers: int = 16):
        super().__init__(audio_dir, metadata_path, output_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.matched_cache = "matched_audiodatafull.json"
        
    def load_metadata(self) -> pd.DataFrame:
        """Load metadata from CSV."""
        print(f"Loading metadata from {self.audio_dir}")
        
        csv_path = self.audio_dir / "df_lemm_srl_path.csv"
        
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                print(f"  Loaded {len(df)} entries from {csv_path.name}")
                
                # Convert paths to local paths
                if 'path' in df.columns:
                    df['local_path'] = df['path'].str.replace(
                        '/root/share/AudioData/', 
                        str(self.audio_dir) + '/',
                        regex=False
                    )
                
                return df
            except Exception as e:
                print(f"  Error loading {csv_path.name}: {e}")
        
        print("No metadata CSV file found")
        return pd.DataFrame()
    
    def get_matched(self, metadata_df: pd.DataFrame) -> Tuple[List, List, List]:
        """Process files based on metadata."""
        print(f"  Processing {len(metadata_df)} files from metadata")
        
        audio_paths = []
        metadata_list = []
        
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            try:
                audio_path = Path(row['local_path'])
                
                if not audio_path.exists():
                    continue
                
                metadata = {
                    'split': 'train',
                    'original_filename': row['filename'],
                    'full_path': row.get('path', '').replace('/root/share/AudioData/', ''),
                    'original_description': row.get('description', '')
                }
                
                audio_paths.append(audio_path)
                metadata_list.append(metadata)
                                        
            except Exception as e:
                print(f"    Error processing row {idx}: {e}")
                continue
        
        print(f"\n  Found {len(audio_paths)} valid audio files")
        
        # Create text descriptions from metadata
        texts = []
        for metadata in metadata_list:
            desc_parts = []
            if metadata.get('original_filename'):
                desc_parts.append(f"Filename: {metadata['original_filename']}")
            if metadata.get('original_description'):
                desc_parts.append(f"Description: {metadata['original_description']}")
            texts.append(" | ".join(desc_parts) if desc_parts else "")
        
        return audio_paths, texts, metadata_list
        
    def match_audio_to_text(self, metadata_df: pd.DataFrame) -> List[Tuple[Path, str, Dict]]:
        """Process audio files using metadata."""
        matched = []
        
        print(f"Processing audio files from {self.audio_dir}")
        
        # Check cache
        if os.path.exists(self.matched_cache):
            print(f"  Loading cached matched pairs from {self.matched_cache}")
            with open(self.matched_cache, "r") as f:
                cached = json.load(f)
                audio_paths = [Path(p) for p, _, _ in cached]
                texts = [t for _, t, _ in cached]
                metadata_list = [m for _, _, m in cached]
        else:
            print(f"  No cache found, processing metadata")
            audio_paths, texts, metadata_list = self.get_matched(metadata_df)
            
            # Save cache
            with open(self.matched_cache, "w") as f:
                cached = [(str(p), t, m) for p, t, m in zip(audio_paths, texts, metadata_list)]
                json.dump(cached, f)
        
        # Create matched list
        for audio_path, text, metadata in zip(audio_paths, texts, metadata_list):
            matched.append((audio_path, text, metadata))
                            
        print(f"Total matched: {len(matched)} audio-text pairs")
        return matched
    
    def process_dataset(self, samples_per_tar: int = 2048, server_url: str = "http://127.0.0.1:30000"):
        """Process dataset using SGLang hybrid streaming."""
        # Load metadata
        metadata_df = self.load_metadata()
        
        # Match audio to text
        matched_samples = self.match_audio_to_text(metadata_df)
        
        # Group by split
        samples_by_split = {}
        for audio_path, text, metadata in matched_samples:
            split = metadata.get('split', 'train')
            if split not in samples_by_split:
                samples_by_split[split] = []
            samples_by_split[split].append((audio_path, text, metadata))
        
        # Process each split
        all_stats = {}
        for split, split_samples in samples_by_split.items():
            print(f"\nProcessing {split} split with {len(split_samples)} samples...")
            
            # Create hybrid streaming processor
            processor = HybridStreamingProcessor(
                output_dir=self.output_dir,
                model_name="google/gemma-3n-e4b-it",
                system_prompt="You are a helpful assistant.",
                user_prompt_template="Describe in detail what you hear in the audio in max. 80 words: <start_of_audio><audio_soft_token><end_of_audio> . For context, use the following information but don't quote it: {text}",
                generation_config={
                    "max_new_tokens": 200,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
                samples_per_tar=samples_per_tar,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sglang_server_url=server_url
            )
            
            # Process with hybrid streaming
            stats = processor.process_dataset(
                audio_files=split_samples,
                prefix=self.__class__.__name__.lower().replace('processor', ''),
                split=split,
                show_progress=True
            )
            
            all_stats[split] = stats
        
        # Summary
        total_processed = sum(s['written'] for s in all_stats.values())
        total_failed = sum(s['failed'] for s in all_stats.values())
        
        print(f"\nProcessing complete!")
        print(f"Total samples attempted: {len(matched_samples)}")
        print(f"Successfully processed: {total_processed}")
        print(f"Failed: {total_failed}")
        
        return all_stats


def main():
    parser = argparse.ArgumentParser(description="Process AudioDataFull with SGLang")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/AudioDataFull/",
                       help="Path to AudioDataFull directory")
    parser.add_argument("--metadata", type=str,
                       default="/scratch-shared/gwijngaard/laion/AudioDataFull/",
                       help="Path to metadata directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/audiodatafull",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=2048,
                       help="Number of samples per tar file")
    parser.add_argument("--batch-size", type=int, default=128,
                       help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=16,
                       help="Number of workers for data loading")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:30000",
                       help="SGLang server URL")
    
    args = parser.parse_args()
    
    # Create processor
    processor = AudioDataFullProcessor(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Process dataset
    processor.process_dataset(
        samples_per_tar=args.samples_per_tar,
        server_url=args.server_url
    )


if __name__ == "__main__":
    main()