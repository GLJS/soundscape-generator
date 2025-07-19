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
        
    def create_caption_from_events(self, events: List[Dict], start_time: float, end_time: float) -> str:
        """Create a natural language caption from sound events in a time segment."""
        # Filter events that overlap with the segment
        segment_events = defaultdict(float)
        
        for event in events:
            # Check if event overlaps with segment
            if event['start_time'] < end_time and event['end_time'] > start_time:
                # Calculate overlap duration
                overlap_start = max(event['start_time'], start_time)
                overlap_end = min(event['end_time'], end_time)
                overlap_duration = overlap_end - overlap_start
                
                # Weight by confidence and overlap duration
                weight = event['confidence'] * (overlap_duration / (end_time - start_time))
                segment_events[event['event']] += weight
        
        # Sort events by weight and create caption
        if segment_events:
            sorted_events = sorted(segment_events.items(), key=lambda x: x[1], reverse=True)
            # Take top events with significant presence (weight > 0.1)
            significant_events = [event for event, weight in sorted_events if weight > 0.1]
            
            if significant_events:
                if len(significant_events) == 1:
                    return f"Sound of {significant_events[0]}"
                elif len(significant_events) == 2:
                    return f"Sounds of {significant_events[0]} and {significant_events[1]}"
                else:
                    events_str = ", ".join(significant_events[:-1]) + f" and {significant_events[-1]}"
                    return f"Sounds of {events_str}"
        
        return "Ambient sounds"
        
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
                caption = self.create_caption_from_events(annotations, start_time, end_time)
                
                # Add scene context to caption
                scene_name = scene.replace('_', ' ')
                full_caption = f"{scene_name} with {caption.lower()}"
                
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
        
    def match_audio_to_text(self, metadata_df=None) -> List[Tuple[bytes, str, Dict]]:
        """Process all audio files and create segments with captions."""
        all_segments = []
        
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
                    all_segments.extend(segments)
                    print(f"  Processed {audio_file.name}: {len(segments)} segments")
                else:
                    print(f"  No annotations found for {audio_file.name}")
                    
        print(f"\nTotal segments created: {len(all_segments)}")
        return all_segments
        


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