# LAION Audio Dataset Conversion Scripts - Complete Collection

This directory contains conversion scripts for ALL datasets in `/scratch-shared/gwijngaard/laion/`.

## Dataset Scripts

### Datasets with Metadata (High Priority)

1. **animalspeak.py** - 1.09 TB, animal sounds with species information
2. **audiodiffcaps.py** - Paired audio with differential captions
3. **audiohallucination.py** - Test dataset with adversarial/popular/random subsets
4. **autoacd.py** - 778 GB FLAC archive
5. **clothochatgptmixup.py** - Mixed audio with GPT-generated captions
6. **clothodetail.py** - Detailed Clotho annotations
7. **clothoentailment.py** - Entailment/neutral/contradiction captions
8. **clothomoment.py** - Temporal audio captions
9. **epidemic.py** - 151k sound effects
10. **esc50.py** - Environmental sound classification (50 classes)
11. **favdbench.py** - Audio benchmark dataset
12. **lass.py** - Language-audio similarity dataset
13. **macs.py** - Multi-annotated captions
14. **maqa.py** - Multilingual audio QA
15. **richdetailaudio.py** - Rich detail audio-text simulation
16. **soundbible.py** - Sound effects with descriptions
17. **sounddescs.py** - 555 GB with categories and descriptions
18. **soundingearth.py** - 403 GB nature/environment sounds
19. **soundjay.py** - Sound effects library
20. **texttoaudiogrounding.py** - Temporal grounding dataset
21. **vggsound.py** - 200k+ YouTube clips (20 archives)

### Datasets without Metadata in data/ folder

22. **audiodatafull.py** - 363 GB multi-source collection
23. **audiolog.py** - Empty directory placeholder
24. **avqa.py** - Audio-visual QA (empty)
25. **compa.py** - Musical instrument solos
26. **epickitchens.py** - Kitchen activities (HDF5 format)
27. **gise51.py** - Sound event detection dataset
28. **laion630k.py** - Main LAION-630k aggregated dataset
29. **recap.py** - Re-captioned audio dataset
30. **soundscaper.py** - Soundscape generation
31. **urbansed.py** - Urban sound event detection

### Generic Script

32. **generic_dataset.py** - Use for any dataset not listed above

## Usage

Each script can be run independently:

```bash
# Basic usage
python {dataset_name}.py

# With custom parameters
python {dataset_name}.py --output-dir /custom/path --samples-per-tar 512

# Example for VGGSound
python vggsound.py --audio-dir /scratch-shared/gwijngaard/laion/VGGSound \
                   --output-dir /scratch-shared/gwijngaard/tar/vggsound

# Generic script for unlisted datasets
python generic_dataset.py --audio-dir /path/to/dataset \
                         --output-dir /path/to/output \
                         --dataset-name mydataset
```

## Common Parameters

- `--audio-dir`: Path to audio files/archives (default: dataset-specific)
- `--metadata`: Path to metadata files (default: dataset-specific)
- `--output-dir`: Output directory for tar files (default: `/scratch-shared/gwijngaard/tar/{dataset}`)
- `--samples-per-tar`: Number of samples per tar file (default: 256)

## Output Structure

All scripts create WebDataset-compatible tar files:
```
/scratch-shared/gwijngaard/tar/{dataset}/
├── {dataset}_000000.tar
├── {dataset}_000001.tar
├── ...
└── sizes.json
```

## Special Cases

- **VGGSound**: Processes 20 separate tar.gz archives
- **Epic-Kitchens**: Handles HDF5 format audio
- **LAION-630k**: Aggregates multiple sub-datasets, use `--dataset-filter` to process specific ones
- **AudioDataFull**: Large dataset (363GB), processes in streaming mode with `--limit` option

## Adding New Datasets

1. Check if the dataset exists in `/scratch-shared/gwijngaard/laion/`
2. If not covered by existing scripts, use `generic_dataset.py`
3. For complex datasets, copy a similar script and modify

## Dependencies

- See `../utils.py` for common functionality
- Requires: pandas, librosa, soundfile, tqdm, pyyaml
- Some datasets need additional: h5py (epic-kitchens), tarfile, zipfile

## Notes

- Scripts handle missing files gracefully
- Audio is converted to FLAC 48kHz mono 16-bit by default
- Progress bars show processing status
- Errors are logged but don't stop processing

## Directory Mapping

The scripts automatically handle various archive formats:
- `.tar.gz`, `.tar.bz2` - Compressed tar archives
- `.zip` - ZIP archives  
- `.tar` - Uncompressed tar archives
- Direct directories with audio files

Each script is optimized for its specific dataset structure and metadata format.