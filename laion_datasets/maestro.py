#!/usr/bin/env python3
"""
Convert MAESTRO dataset to WebDataset tar format.

Audio location: /scratch-shared/gwijngaard/laion/MAESTRO/development_audio/
Annotations: /scratch-shared/gwijngaard/laion/MAESTRO/development_annotation/
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from utils import DatasetProcessor, AudioProcessor, TarCreator
from typing import List, Tuple, Dict
import argparse
import librosa
import soundfile as sf
import io
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()


class MAESTROProcessor(DatasetProcessor):
    """Processor for MAESTRO dataset with soft-labeled sound events."""
    
    def __init__(self, audio_dir: str, annotation_dir: str, output_dir: str):
        super().__init__(audio_dir, "", output_dir)
        self.annotation_dir = Path(annotation_dir)
        self.segment_duration = 30  # 30 seconds per segment
        self.scenes = ['cafe_restaurant', 'city_center', 'grocery_store', 
                      'metro_station', 'residential_area']
    
    def load_metadata(self) -> pd.DataFrame:
        """Return empty DataFrame as MAESTRO uses per-file annotations."""
        return pd.DataFrame()
        
    def load_annotations(self, scene: str, audio_file: str) -> List[Dict]:
        """Load soft label annotations for a specific audio file."""
        annotations = []
        
        # Construct annotation file path
        base_name = audio_file.replace('.wav', '')
        annotation_file = self.annotation_dir / f"soft_labels_{scene}" / f"{base_name}.txt"
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 4:
                        annotations.append({
                            'start_time': float(parts[0]),
                            'end_time': float(parts[1]),
                            'event': parts[2],
                            'confidence': float(parts[3])
                        })
        
        return annotations
        
    def create_caption_from_events(self, events: List[Dict], start_time: float, end_time: float, scene: str) -> str:
        """Create a natural language caption from sound events in a time segment."""
        # Filter events that overlap with the segment and have confidence >= 0.5
        segment_events = set()
        
        for event in events:
            # Check if event overlaps with segment and meets confidence threshold
            if (event['start_time'] < end_time and event['end_time'] > start_time 
                and event['confidence'] >= 0.5):
                segment_events.add(event['event'])
            
        scene = scene.replace('_', ' ')
        
        # Create caption from unique events
        if segment_events:
            events_list = sorted(list(segment_events))
            if len(events_list) == 1:
                return f"There is {events_list[0]} in a {scene}"
            elif len(events_list) == 2:
                return f"There is {events_list[0]} and {events_list[1]} in a {scene}"
            else:
                # Join all but last with commas, then add 'and' before the last
                events_str = ", ".join(events_list[:-1]) + f" and {events_list[-1]}"
                return f"There is {events_str} in a {scene}"
        
        return None
        
    def segment_audio(self, audio_path: Path, annotations: List[Dict]) -> List[Tuple[bytes, str, Dict]]:
        """Segment audio file into 30-second chunks with captions."""
        segments = []
        
        try:
            # Load full audio
            audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            
            # Calculate total duration and number of segments
            total_duration = len(audio) / sr
            num_segments = int(total_duration / self.segment_duration)
            
            # Extract scene type from path
            scene = audio_path.parent.name
            
            for i in range(num_segments):
                start_time = i * self.segment_duration
                end_time = min((i + 1) * self.segment_duration, total_duration)
                
                # Extract audio segment
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                segment_audio = audio[start_sample:end_sample]
                
                # Resample to 48kHz if needed
                if sr != 48000:
                    segment_audio = librosa.resample(segment_audio, orig_sr=sr, target_sr=48000)
                
                # Create caption from events
                caption = self.create_caption_from_events(annotations, start_time, end_time, scene)
                if caption is None:
                    continue
                
                # Use caption as-is (already includes "in the background")
                full_caption = caption
                
                # Convert to FLAC bytes
                output_buffer = io.BytesIO()
                sf.write(output_buffer, segment_audio, 48000, format='FLAC', subtype='PCM_16')
                output_buffer.seek(0)
                audio_bytes = output_buffer.read()
                
                # Create metadata
                # Since this is development_audio, set split to 'train'
                metadata = {
                    'split': 'train',
                    'scene': scene,
                    'original_filename': audio_path.name,
                    'segment_index': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'sample_rate': 48000,
                    'channels': 1,
                    'format': 'flac',
                    'task': 'AAC'
                }
                
                # Add event details to metadata
                segment_events = []
                for event in annotations:
                    if event['start_time'] < end_time and event['end_time'] > start_time:
                        segment_events.append({
                            'event': event['event'],
                            'confidence': event['confidence']
                        })
                metadata['events'] = segment_events
                
                segments.append((audio_bytes, full_caption, metadata))
                
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")
            
        return segments
        
    def process_dataset(self, samples_per_tar: int = 256):
        """Override process_dataset to handle MAESTRO's unique structure."""
        all_samples = []
        
        # Process each scene
        for scene in self.scenes:
            scene_dir = self.audio_dir / scene
            if not scene_dir.exists():
                print(f"Scene directory not found: {scene_dir}")
                continue
                
            print(f"\nProcessing scene: {scene}")
            audio_files = sorted(scene_dir.glob("*.wav"))
            
            for audio_file in audio_files:
                # Load annotations
                annotations = self.load_annotations(scene, audio_file.name)
                
                if annotations:
                    # Segment audio and create captions
                    segments = self.segment_audio(audio_file, annotations)
                    
                    # Convert segments to the format expected by TarCreator
                    for audio_bytes, caption, metadata in segments:
                        all_samples.append({
                            'audio_bytes': audio_bytes,
                            'text': caption,
                            'metadata': metadata
                        })
                    
                    print(f"  Processed {audio_file.name}: {len(segments)} segments")
                else:
                    print(f"  No annotations found for {audio_file.name}")
        
        print(f"\nTotal segments created: {len(all_samples)}")
        
        # Create tar files
        tar_creator = TarCreator(self.output_dir, prefix='maestro', 
                                samples_per_tar=samples_per_tar, split='train')
        
        # Process in batches
        all_summaries = []
        for i in range(0, len(all_samples), samples_per_tar):
            batch = all_samples[i:i+samples_per_tar]
            if batch:
                summary = tar_creator.create_tar_from_samples(batch, i // samples_per_tar)
                all_summaries.append(summary)
        
        # Create size file
        tar_creator.create_size_file(all_summaries)
        
        # Print summary
        print(f"\nDataset processing complete!")
        print(f"Created {len(all_summaries)} tar files")
        total_successful = sum(s['successful'] for s in all_summaries)
        total_failed = sum(s['failed'] for s in all_summaries)
        print(f"Total successful: {total_successful}")
        print(f"Total failed: {total_failed}")
    
    def match_audio_to_text(self, metadata_df=None) -> List[Tuple[bytes, str, Dict]]:
        """Not used in MAESTRO - see process_dataset instead."""
        raise NotImplementedError("MAESTRO uses custom process_dataset implementation")
        


def main():
    parser = argparse.ArgumentParser(description="Convert MAESTRO dataset to tar format")
    parser.add_argument("--audio-dir", type=str, 
                       default="/scratch-shared/gwijngaard/laion/MAESTRO/development_audio",
                       help="Path to audio directory")
    parser.add_argument("--annotation-dir", type=str,
                       default="/scratch-shared/gwijngaard/laion/MAESTRO/development_annotation",
                       help="Path to annotation directory")
    parser.add_argument("--output-dir", type=str,
                       default="/scratch-shared/gwijngaard/tar/maestro",
                       help="Output directory for tar files")
    parser.add_argument("--samples-per-tar", type=int, default=256,
                       help="Number of samples per tar file")
    
    args = parser.parse_args()
    
    # Create processor
    processor = MAESTROProcessor(
        audio_dir=args.audio_dir,
        annotation_dir=args.annotation_dir,
        output_dir=args.output_dir
    )
    
    # Process dataset
    processor.process_dataset(samples_per_tar=args.samples_per_tar)


if __name__ == "__main__":
    main()