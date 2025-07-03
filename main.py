#!/usr/bin/env python3
"""
Main execution script for LAION data preparation and soundscape generation.
This script orchestrates the complete pipeline from data loading to soundscape creation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

from data_loader import LAIONDataLoader
from soundscape_generator import SoundscapeGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='LAION Data Preparation and Soundscape Generation Pipeline')
    
    # Data loading arguments
    parser.add_argument('--data-root', default='/scratch-shared/gwijngaard/laion/',
                       help='Root directory containing LAION datasets')
    parser.add_argument('--dataset', default='generated-sound-events',
                       choices=['generated-sound-events', 'audiosnippets', 'freesound', 
                               'audioset', 'music', 'wild-events'],
                       help='Dataset to use for soundscape generation')
    parser.add_argument('--cache-dir', default='/scratch-shared/gwijngaard/laion/extracted/cache',
                       help='Directory to cache file indices and metadata')
    parser.add_argument('--extract-dir', default='/scratch-shared/gwijngaard/laion/extracted/data',
                       help='Directory to extract tar files')
    
    # Soundscape generation arguments
    parser.add_argument('--output-dir', default='/scratch-shared/gwijngaard/laion/extracted/output',
                       help='Directory to save generated soundscapes')
    parser.add_argument('--num-soundscapes', type=int, default=10,
                       help='Number of soundscapes to generate')
    parser.add_argument('--min-samples', type=int, default=2,
                       help='Minimum number of audio samples per soundscape')
    parser.add_argument('--max-samples', type=int, default=5,
                       help='Maximum number of audio samples per soundscape')
    parser.add_argument('--min-duration', type=float, default=2.0,
                       help='Minimum duration of each event in seconds')
    parser.add_argument('--max-duration', type=float, default=30.0,
                       help='Maximum total duration of soundscape in seconds')
    
    # Pipeline control arguments
    parser.add_argument('--skip-extraction', action='store_true',
                       help='Skip data extraction (use existing extracted data)')
    parser.add_argument('--only-extract', action='store_true',
                       help='Only extract data, do not generate soundscapes')
    parser.add_argument('--max-workers', type=int,
                       help='Maximum number of parallel workers for extraction')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    Path(args.extract_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path('./logs').mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting LAION Data Preparation and Soundscape Generation Pipeline")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Step 1: Initialize data loader
        logger.info("Initializing data loader...")
        loader = LAIONDataLoader(
            data_root=args.data_root,
            cache_dir=args.cache_dir,
            extract_dir=args.extract_dir
        )
        
        # Step 2: Discover available datasets
        logger.info("Discovering available datasets...")
        datasets = loader.discover_datasets()
        
        if not datasets:
            logger.error("No datasets found in the data root directory")
            return 1
        
        logger.info("Available datasets:")
        for name, files in datasets.items():
            logger.info(f"  {name}: {len(files)} tar files")
        
        if args.dataset not in datasets:
            logger.error(f"Dataset '{args.dataset}' not found in available datasets")
            return 1
        
        # Step 3: Extract dataset (if not skipped)
        if not args.skip_extraction:
            logger.info(f"Extracting dataset: {args.dataset}")
            success = loader.extract_dataset(args.dataset, max_workers=args.max_workers)
            
            if not success:
                logger.error(f"Failed to extract dataset: {args.dataset}")
                return 1
            
            logger.info("Dataset extraction completed successfully")
        else:
            logger.info("Skipping data extraction (using existing data)")
        
        # Step 4: Load file index
        logger.info("Loading file index...")
        file_index = loader.load_file_index(args.dataset)
        
        if not file_index:
            logger.error(f"No file index found for dataset: {args.dataset}")
            logger.info("Try running without --skip-extraction to extract the dataset first")
            return 1
        
        # Step 5: Get dataset statistics
        logger.info("Getting dataset statistics...")
        stats = loader.get_dataset_stats(args.dataset)
        logger.info(f"Dataset statistics: {stats}")
        
        # Step 6: Filter files for soundscape generation
        logger.info("Filtering files for soundscape generation...")
        suitable_files = loader.get_files_by_criteria(
            args.dataset,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            has_caption=True
        )
        
        if not suitable_files:
            logger.error("No suitable files found for soundscape generation")
            logger.info("Try adjusting the duration criteria or use a different dataset")
            return 1
        
        logger.info(f"Found {len(suitable_files)} files suitable for soundscape generation")
        
        # Stop here if only extraction was requested
        if args.only_extract:
            logger.info("Extraction completed. Exiting as requested.")
            return 0
        
        # Step 7: Initialize soundscape generator
        logger.info("Initializing soundscape generator...")
        generator = SoundscapeGenerator(output_dir=args.output_dir)
        
        # Update generator configuration
        generator.NUM_SAMPLES_TO_GENERATE = args.num_soundscapes
        generator.MIN_SAMPLES_PER_SOUNDSCAPE = args.min_samples
        generator.MAX_SAMPLES_PER_SOUNDSCAPE = args.max_samples
        generator.MIN_EVENT_DURATION_S = args.min_duration
        generator.MAX_TOTAL_DURATION_S = args.max_duration
        
        # Step 8: Find starting index for output files
        existing_files = list(Path(args.output_dir).glob("*.mp3"))
        if existing_files:
            existing_indices = [int(f.stem) for f in existing_files if f.stem.isdigit()]
            start_index = max(existing_indices) + 1 if existing_indices else 0
        else:
            start_index = 0
        
        logger.info(f"Starting soundscape generation from index {start_index}")
        
        # Step 9: Generate soundscapes
        logger.info("Starting soundscape generation...")
        success_count = generator.generate_soundscapes(suitable_files, start_index)
        
        if success_count > 0:
            logger.info(f"Successfully generated {success_count}/{args.num_soundscapes} soundscapes")
            logger.info(f"Output files saved to: {args.output_dir}")
        else:
            logger.error("Failed to generate any soundscapes")
            return 1
        
        logger.info("Pipeline completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)