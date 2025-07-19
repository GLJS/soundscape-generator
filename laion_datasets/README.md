# LAION Audio Dataset Conversion Scripts

This directory contains scripts to convert various audio datasets from the LAION collection into WebDataset tar format.

## Overview

Each script processes a specific dataset by:
1. Loading metadata/captions from `/gpfs/work4/0/einf6190/data-preparation/data/`
2. Loading audio files from `/scratch-shared/gwijngaard/laion/`
3. Converting audio to FLAC format (48kHz, mono, 16-bit)
4. Creating WebDataset-compatible tar files in `/scratch-shared/gwijngaard/tar/`

## Common Structure

All scripts follow the same pattern:
- Audio files are paired with text captions
- Output format: `{key}.flac` and `{key}.json` pairs in tar files
- Default: 256 samples per tar file
- Includes `sizes.json` for WebDataset compatibility

## Available Scripts

### 1. **animalspeak.py**
- **Audio**: MP3, WAV, M4A files in `audio/` directory
- **Size**: 1.09 TB
- **Metadata**: AnimalSpeak_correct.csv
- **Usage**: `python animalspeak.py`

### 2. **audiodiffcaps.py**
- **Audio**: WAV files in `data/` directory
- **Metadata**: Multiple CSV files in `csv/` directory
- **Note**: Contains paired audio files with differential captions
- **Usage**: `python audiodiffcaps.py`

### 3. **clothochatgptmixup.py**
- **Audio**: Compressed in `audio.tar.gz`
- **Metadata**: mixed_audio_info.csv
- **Note**: Extracts from archive on-the-fly
- **Usage**: `python clothochatgptmixup.py`

### 4. **esc50.py**
- **Audio**: Compressed in `audio.tar.gz`
- **Metadata**: esc50.csv
- **Note**: Environmental sound classification dataset
- **Usage**: `python esc50.py`

### 5. **vggsound.py**
- **Audio**: 20 tar.gz files (`vggsound_00.tar.gz` to `vggsound_19.tar.gz`)
- **Metadata**: vggsound.csv
- **Note**: Large-scale dataset, processes multiple archives
- **Usage**: `python vggsound.py`

### 6. **autoacd.py**
- **Audio**: FLAC files in `flac/` or `flac.tar.bz2`
- **Size**: 778 GB archive
- **Metadata**: train.csv and test.csv
- **Usage**: `python autoacd.py`

## Command Line Arguments

All scripts support the following arguments:
- `--audio-dir`: Path to audio files (default: dataset-specific)
- `--metadata`: Path to metadata files (default: dataset-specific)
- `--output-dir`: Output directory for tar files (default: `/scratch-shared/gwijngaard/tar/{dataset}`)
- `--samples-per-tar`: Number of samples per tar file (default: 256)

## Example Usage

```bash
# Process AnimalSpeak dataset
python animalspeak.py

# Process ESC50 with custom output directory
python esc50.py --output-dir /custom/path/esc50

# Process VGGSound with more samples per tar
python vggsound.py --samples-per-tar 512
```

## Output Format

Each tar file contains:
```
000000_000000.flac  # Audio file (FLAC, 48kHz, mono, 16-bit)
000000_000000.json  # Metadata and caption
000000_000001.flac
000000_000001.json
...
```

JSON format:
```json
{
  "text": "A dog barking",
  "caption": "A dog barking",
  "split": "train",
  "original_filename": "1234.mp3",
  "sample_rate": 48000,
  "duration": 5.2,
  "channels": 1,
  "format": "flac"
}
```

## Adding New Datasets

To add a new dataset:

1. Create a new script following the pattern of existing ones
2. Inherit from `DatasetProcessor` in `utils.py`
3. Implement:
   - `load_metadata()`: Load CSV/JSON metadata
   - `match_audio_to_text()`: Match audio files to captions

## Requirements

- Python 3.7+
- See parent directory's requirements.txt for dependencies
- Sufficient disk space for output tar files

## Notes

- Scripts handle missing audio files gracefully
- Progress is shown with tqdm progress bars
- Errors are logged but don't stop processing
- Audio is converted to consistent format for training