#!/usr/bin/env python3
"""
Enrich SoundingEarth metadata with location information using GIS reverse geocoding.
This script adds 'location' and 'country' columns to the metadata.csv file.
"""

import pandas as pd
from pathlib import Path
import time
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Import geopy for reverse geocoding
try:
    from geopy.geocoders import Nominatim
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
except ImportError:
    print("Error: geopy is not installed. Please install it with: pip install geopy")
    exit(1)


def get_location_from_coords(lat, lon, geolocator, cache=None):
    """
    Get location information from latitude and longitude coordinates.
    Returns (location, country) tuple.
    """
    # Check cache first
    cache_key = f"{lat:.6f},{lon:.6f}"
    if cache and cache_key in cache:
        return cache[cache_key]
    
    try:
        # Reverse geocode
        location = geolocator.reverse(f"{lat}, {lon}", exactly_one=True, language='en')
        
        if location and location.raw:
            address = location.raw.get('address', {})
            
            # Extract location (prefer city, then town, then village, then suburb)
            location_name = (
                address.get('city') or 
                address.get('town') or 
                address.get('village') or 
                address.get('suburb') or
                address.get('municipality') or
                address.get('county') or
                address.get('state') or
                'Unknown'
            )
            
            # Extract country
            country = address.get('country', 'Unknown')
            
            result = (location_name, country)
            
            # Cache the result
            if cache is not None:
                cache[cache_key] = result
            
            return result
        else:
            return ('Unknown', 'Unknown')
            
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geocoding error for {lat}, {lon}: {e}")
        return ('Unknown', 'Unknown')
    except Exception as e:
        print(f"Unexpected error for {lat}, {lon}: {e}")
        return ('Unknown', 'Unknown')


def enrich_metadata(metadata_path, output_path, cache_file='gis_cache.json'):
    """
    Enrich metadata CSV with location and country information.
    """
    # Load existing metadata
    print(f"Loading metadata from {metadata_path}")
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} entries")
    
    # Initialize geocoder with a user agent
    geolocator = Nominatim(user_agent="soundingearth_metadata_enrichment")
    
    # Load cache if exists
    cache = {}
    cache_path = Path(cache_file)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            print(f"Loaded {len(cache)} cached locations")
        except:
            print("Could not load cache, starting fresh")
    
    # Check if enrichment columns already exist
    if 'location' in df.columns and 'country' in df.columns:
        print("Location and country columns already exist. Filling missing values only.")
        missing_mask = df['location'].isna() | df['country'].isna()
        rows_to_process = df[missing_mask]
    else:
        # Add new columns
        df['location'] = None
        df['country'] = None
        rows_to_process = df
    
    print(f"Processing {len(rows_to_process)} rows...")
    
    # Process each row with coordinates
    processed_count = 0
    for idx, row in tqdm(rows_to_process.iterrows(), total=len(rows_to_process)):
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            location, country = get_location_from_coords(
                row['latitude'], 
                row['longitude'], 
                geolocator,
                cache
            )
            
            df.at[idx, 'location'] = location
            df.at[idx, 'country'] = country
            processed_count += 1
            
            # Rate limiting - Nominatim requires 1 second between requests
            time.sleep(1.0)
            
            # Save cache periodically
            if processed_count % 50 == 0:
                with open(cache_path, 'w') as f:
                    json.dump(cache, f)
                print(f"Saved cache with {len(cache)} entries")
    
    # Save final cache
    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f"Final cache saved with {len(cache)} entries")
    
    # Save enriched metadata
    print(f"Saving enriched metadata to {output_path}")
    df.to_csv(output_path, index=False)
    print(f"Enriched {processed_count} entries with location information")
    
    # Print statistics
    location_counts = df['location'].value_counts()
    country_counts = df['country'].value_counts()
    
    print("\nTop 10 locations:")
    print(location_counts.head(10))
    
    print("\nTop 10 countries:")
    print(country_counts.head(10))
    
    unknown_count = (df['location'] == 'Unknown').sum()
    print(f"\nEntries with unknown location: {unknown_count}")


def main():
    parser = argparse.ArgumentParser(description="Enrich SoundingEarth metadata with GIS location data")
    parser.add_argument("--metadata", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/SoundingEarth/metadata.csv",
                       help="Path to original metadata.csv")
    parser.add_argument("--output", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/SoundingEarth/metadata_enriched.csv",
                       help="Path to save enriched metadata")
    parser.add_argument("--cache", type=str,
                       default="/gpfs/work4/0/einf6190/data-preparation/data/SoundingEarth/gis_cache.json",
                       help="Path to cache file for geocoding results")
    
    args = parser.parse_args()
    
    # Ensure metadata exists
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: Metadata file not found at {metadata_path}")
        exit(1)
    
    # Run enrichment
    enrich_metadata(metadata_path, args.output, args.cache)


if __name__ == "__main__":
    main()