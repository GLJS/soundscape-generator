#!/usr/bin/env python3
"""
RAR File Extraction Script for Nested Archives

This script finds and extracts all RAR files within the extracted directory,
preserving their path structure and internal file organization.

Usage: python extract_archives.py
"""

import os
import rarfile
import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Directory path
EXTRACT_DIR = "/gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx/extracted"

def find_rar_files(base_dir: str) -> List[str]:
    """
    Find all RAR files in the directory tree using os.walk.
    
    Args:
        base_dir: Base directory to search in
        
    Returns:
        List of absolute paths to RAR files
    """
    rar_files = []
    
    if not os.path.exists(base_dir):
        logger.error(f"Directory does not exist: {base_dir}")
        return rar_files
    
    logger.info(f"Scanning directory: {base_dir}")
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.rar'):
                file_path = os.path.join(root, file)
                rar_files.append(file_path)
                logger.debug(f"Found RAR: {file_path}")
    
    logger.info(f"Found {len(rar_files)} RAR files")
    return rar_files

def get_extraction_path(rar_path: str) -> str:
    """
    Get the extraction path for a RAR file.
    The RAR will be extracted to a subdirectory with the same name (without .rar)
    in the same location as the RAR file.
    
    Args:
        rar_path: Path to the RAR file
        
    Returns:
        Path where the RAR should be extracted
    """
    # Get directory and filename
    dir_path = os.path.dirname(rar_path)
    filename = os.path.basename(rar_path)
    
    # Remove .rar extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Create extraction path
    extraction_path = os.path.join(dir_path, name_without_ext)
    
    return extraction_path

def extract_rar_file(rar_path: str, extract_to: str) -> bool:
    """
    Extract a RAR file to the specified directory.
    
    Args:
        rar_path: Path to the RAR file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create extraction directory
        Path(extract_to).mkdir(parents=True, exist_ok=True)
        
        with rarfile.RarFile(rar_path, 'r') as rar_ref:
            # Extract all files while preserving directory structure
            rar_ref.extractall(extract_to)
            
        logger.info(f"Successfully extracted: {rar_path}")
        logger.info(f"  → Extracted to: {extract_to}")
        return True
        
    except rarfile.BadRarFile:
        logger.error(f"Bad RAR file: {rar_path}")
        return False
    except PermissionError:
        logger.error(f"Permission denied: {rar_path}")
        return False
    except Exception as e:
        logger.error(f"Error extracting {rar_path}: {str(e)}")
        return False

def extract_all_rars():
    """Main function to extract all RAR files."""
    logger.info("Starting RAR extraction process")
    logger.info(f"Working directory: {EXTRACT_DIR}")
    
    # Find all RAR files
    rar_files = find_rar_files(EXTRACT_DIR)
    
    if not rar_files:
        logger.warning("No RAR files found to extract")
        return
    
    successful_extractions = 0
    failed_extractions = 0
    skipped_extractions = 0
    
    for rar_path in rar_files:
        logger.info(f"\nProcessing: {rar_path}")
        
        # Get extraction path
        extraction_path = get_extraction_path(rar_path)
        
        # Check if already extracted
        if os.path.exists(extraction_path) and os.listdir(extraction_path):
            logger.info(f"  → Already extracted, skipping: {extraction_path}")
            skipped_extractions += 1
            continue
        
        # Extract the RAR file
        success = extract_rar_file(rar_path, extraction_path)
        
        if success:
            successful_extractions += 1
        else:
            failed_extractions += 1
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("EXTRACTION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total RAR files found: {len(rar_files)}")
    logger.info(f"Successful extractions: {successful_extractions}")
    logger.info(f"Failed extractions: {failed_extractions}")
    logger.info(f"Skipped (already extracted): {skipped_extractions}")
    logger.info(f"Base directory: {EXTRACT_DIR}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    print("RAR File Extraction Script")
    print("=" * 60)
    print(f"Target directory: {EXTRACT_DIR}")
    print("This will extract all RAR files found in the directory tree.")
    print("Each RAR will be extracted to a subdirectory at its location.")
    print("=" * 60)
    
    # Ask for confirmation
    response = input("\nProceed with extraction? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        extract_all_rars()
        print("\nExtraction process completed. Check 'extraction.log' for details.")
    else:
        print("Extraction cancelled.")