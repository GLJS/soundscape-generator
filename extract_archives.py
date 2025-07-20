#!/usr/bin/env python3
"""
Archive Extraction Script

This script extracts all ZIP and RAR files from a source directory into 
organized subdirectories while preserving the internal file structure.

Usage: python extract_archives.py
"""

import os
import zipfile
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

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

# Directory paths
SOURCE_DIR = "/gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx/done"
EXTRACT_DIR = "/gpfs/scratch1/shared/gwijngaard/laion/downloaded_sfx/extracted"

def ensure_directory_exists(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {path}")

def find_archive_files(source_dir: str) -> List[Tuple[str, str]]:
    """
    Find all ZIP and RAR files in the source directory.
    
    Returns:
        List of tuples (file_path, file_type)
    """
    archive_files = []
    
    if not os.path.exists(source_dir):
        logger.error(f"Source directory does not exist: {source_dir}")
        return archive_files
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_lower = file.lower()
            
            if file_lower.endswith('.zip'):
                archive_files.append((file_path, 'zip'))
            elif file_lower.endswith('.rar'):
                archive_files.append((file_path, 'rar'))
    
    logger.info(f"Found {len(archive_files)} archive files")
    return archive_files

def extract_zip_file(zip_path: str, extract_to: str) -> bool:
    """
    Extract a ZIP file to the specified directory.
    
    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all files while preserving directory structure
            zip_ref.extractall(extract_to)
        logger.info(f"Successfully extracted ZIP: {zip_path}")
        return True
    except zipfile.BadZipFile:
        logger.error(f"Bad ZIP file: {zip_path}")
        return False
    except Exception as e:
        logger.error(f"Error extracting ZIP {zip_path}: {str(e)}")
        return False

def extract_rar_file(rar_path: str, extract_to: str) -> bool:
    """
    Extract a RAR file to the specified directory using unrar command.
    
    Args:
        rar_path: Path to the RAR file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Try using unrar command
        cmd = ['unrar', 'x', '-y', rar_path, extract_to + '/']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info(f"Successfully extracted RAR: {rar_path}")
            return True
        else:
            logger.error(f"unrar failed for {rar_path}: {result.stderr}")
            
            # Try alternative: 7z command
            cmd_7z = ['7z', 'x', '-y', f'-o{extract_to}', rar_path]
            result_7z = subprocess.run(cmd_7z, capture_output=True, text=True, timeout=300)
            
            if result_7z.returncode == 0:
                logger.info(f"Successfully extracted RAR with 7z: {rar_path}")
                return True
            else:
                logger.error(f"Both unrar and 7z failed for {rar_path}")
                return False
                
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout while extracting RAR: {rar_path}")
        return False
    except FileNotFoundError:
        logger.error("Neither 'unrar' nor '7z' command found. Please install unrar or p7zip-full")
        return False
    except Exception as e:
        logger.error(f"Error extracting RAR {rar_path}: {str(e)}")
        return False

def get_archive_name_without_extension(file_path: str) -> str:
    """Get the archive filename without extension."""
    filename = os.path.basename(file_path)
    # Remove extension (.zip, .rar, etc.)
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext

def extract_archives():
    """Main function to extract all archives."""
    logger.info("Starting archive extraction process")
    
    # Ensure extraction directory exists
    ensure_directory_exists(EXTRACT_DIR)
    
    # Find all archive files
    archive_files = find_archive_files(SOURCE_DIR)
    
    if not archive_files:
        logger.warning("No archive files found to extract")
        return
    
    successful_extractions = 0
    failed_extractions = 0
    
    for archive_path, archive_type in archive_files:
        logger.info(f"Processing {archive_type.upper()} file: {archive_path}")
        
        # Create subdirectory for this archive
        archive_name = get_archive_name_without_extension(archive_path)
        archive_extract_dir = os.path.join(EXTRACT_DIR, archive_name)
        ensure_directory_exists(archive_extract_dir)
        
        # Extract based on file type
        success = False
        if archive_type == 'zip':
            success = extract_zip_file(archive_path, archive_extract_dir)
        elif archive_type == 'rar':
            success = extract_rar_file(archive_path, archive_extract_dir)
        
        if success:
            successful_extractions += 1
            logger.info(f"✓ Extracted to: {archive_extract_dir}")
        else:
            failed_extractions += 1
            logger.error(f"✗ Failed to extract: {archive_path}")
    
    # Summary
    logger.info(f"\n=== EXTRACTION SUMMARY ===")
    logger.info(f"Total archives found: {len(archive_files)}")
    logger.info(f"Successful extractions: {successful_extractions}")
    logger.info(f"Failed extractions: {failed_extractions}")
    logger.info(f"Extraction directory: {EXTRACT_DIR}")

def check_dependencies():
    """Check if required tools are available."""
    logger.info("Checking dependencies...")
    
    # Check for unrar
    try:
        subprocess.run(['unrar'], capture_output=True)
        logger.info("✓ unrar is available")
    except FileNotFoundError:
        logger.warning("✗ unrar not found")
        
        # Check for 7z as alternative
        try:
            subprocess.run(['7z'], capture_output=True)
            logger.info("✓ 7z is available as alternative for RAR files")
        except FileNotFoundError:
            logger.error("✗ Neither unrar nor 7z found. Install with:")
            logger.error("  sudo apt-get install unrar  # for unrar")
            logger.error("  sudo apt-get install p7zip-full  # for 7z")

if __name__ == "__main__":
    print("Archive Extraction Script")
    print("=" * 50)
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Extract directory: {EXTRACT_DIR}")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Ask for confirmation
    response = input("\nProceed with extraction? (y/N): ").strip().lower()
    if response in ['y', 'yes']:
        extract_archives()
        print("\nExtraction process completed. Check 'extraction.log' for details.")
    else:
        print("Extraction cancelled.") 