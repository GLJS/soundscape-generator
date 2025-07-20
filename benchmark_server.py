#!/usr/bin/env python3
"""
Benchmark script to compare performance between original and optimized servers.
"""
from dotenv import load_dotenv

load_dotenv()
import os
import sys
import time
import statistics
from pathlib import Path
import httpx
import asyncio
from typing import List, Dict
import json

# Configuration
SERVERS = {
    "original": os.getenv("ORIGINAL_SERVER_URL", "http://localhost:8080"),
    "optimized": os.getenv("OPTIMIZED_SERVER_URL", "http://localhost:8081")
}

async def benchmark_server(server_url: str, audio_files: List[Path], 
                          prompt: str = "Describe this audio in detail") -> Dict:
    """Benchmark a single server"""
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Check server health
        try:
            health = await client.get(f"{server_url}/health")
            if health.status_code != 200:
                return {"error": "Server not healthy"}
        except Exception as e:
            return {"error": f"Server not reachable: {e}"}
        
        results = {
            "total_time": 0,
            "processing_times": [],
            "loading_times": [],
            "inference_times": [],
            "cache_hits": 0,
            "prefetch_hits": 0,
            "errors": 0
        }
        
        start_time = time.time()
        
        # Process all files
        for i, audio_file in enumerate(audio_files):
            try:
                # Prepare prefetch paths for next files
                prefetch_paths = []
                for j in range(i + 1, min(i + 3, len(audio_files))):
                    prefetch_paths.append(str(audio_files[j]))
                
                headers = {}
                if prefetch_paths and "optimized" in server_url:
                    headers["X-Prefetch-Paths"] = ",".join(prefetch_paths)
                
                # Send request
                response = await client.post(
                    f"{server_url}/generate",
                    json={
                        "audio_path": str(audio_file),
                        "prompt": prompt,
                        "max_new_tokens": 1024
                    },
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Collect metrics
                    results["processing_times"].append(data.get("processing_time_ms", 0))
                    
                    # These fields only exist in optimized server
                    if "loading_time_ms" in data:
                        results["loading_times"].append(data["loading_time_ms"])
                    if "inference_time_ms" in data:
                        results["inference_times"].append(data["inference_time_ms"])
                    if data.get("cached", False):
                        results["cache_hits"] += 1
                    if data.get("prefetched", False):
                        results["prefetch_hits"] += 1
                else:
                    results["errors"] += 1
                    
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results["errors"] += 1
        
        results["total_time"] = time.time() - start_time
        
        # Get final stats from server
        try:
            stats = await client.get(f"{server_url}/stats")
            if stats.status_code == 200:
                results["server_stats"] = stats.json()
        except:
            pass
        
        return results

def print_results(name: str, results: Dict, num_files: int):
    """Print benchmark results"""
    print(f"\n{'='*60}")
    print(f"{name.upper()} SERVER RESULTS")
    print(f"{'='*60}")
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    
    print(f"Total files processed: {num_files}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Average time per file: {results['total_time']/num_files:.2f} seconds")
    print(f"Errors: {results['errors']}")
    
    if results["processing_times"]:
        print(f"\nProcessing times (ms):")
        print(f"  Average: {statistics.mean(results['processing_times']):.2f}")
        print(f"  Median: {statistics.median(results['processing_times']):.2f}")
        print(f"  Min: {min(results['processing_times']):.2f}")
        print(f"  Max: {max(results['processing_times']):.2f}")
    
    if results["loading_times"]:
        print(f"\nLoading times (ms):")
        print(f"  Average: {statistics.mean(results['loading_times']):.2f}")
        print(f"  Total: {sum(results['loading_times']):.2f}")
    
    if results["inference_times"]:
        print(f"\nInference times (ms):")
        print(f"  Average: {statistics.mean(results['inference_times']):.2f}")
        print(f"  Total: {sum(results['inference_times']):.2f}")
    
    if "server_stats" in results:
        stats = results["server_stats"]
        print(f"\nServer statistics:")
        print(f"  Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
        if "prefetch_hit_rate" in stats:
            print(f"  Prefetch hit rate: {stats['prefetch_hit_rate']:.2%}")

async def main():
    """Run benchmarks"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Audio Flamingo servers")
    parser.add_argument("audio_dir", help="Directory containing audio files")
    parser.add_argument("--num-files", type=int, default=20, 
                        help="Number of files to process")
    parser.add_argument("--server", choices=["original", "optimized", "both"], 
                        default="both", help="Which server to benchmark")
    parser.add_argument("--repeat", type=int, default=1, 
                        help="Number of times to repeat benchmark")
    
    args = parser.parse_args()
    
    # Get audio files
    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"Error: Directory {audio_dir} does not exist")
        return
    
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(list(audio_dir.glob(f"*{ext}")))
    
    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return
    
    # Limit to requested number
    audio_files = audio_files[:args.num_files]
    print(f"Benchmarking with {len(audio_files)} audio files")
    
    # Run benchmarks
    servers_to_test = ["original", "optimized"] if args.server == "both" else [args.server]
    
    for server_name in servers_to_test:
        if server_name not in SERVERS:
            continue
            
        print(f"\nBenchmarking {server_name} server...")
        
        all_results = []
        for i in range(args.repeat):
            if args.repeat > 1:
                print(f"  Run {i+1}/{args.repeat}...")
            
            results = await benchmark_server(
                SERVERS[server_name], 
                audio_files
            )
            all_results.append(results)
        
        # Aggregate results if multiple runs
        if args.repeat > 1:
            aggregated = {
                "total_time": statistics.mean([r["total_time"] for r in all_results]),
                "processing_times": [],
                "loading_times": [],
                "inference_times": [],
                "cache_hits": sum(r.get("cache_hits", 0) for r in all_results),
                "prefetch_hits": sum(r.get("prefetch_hits", 0) for r in all_results),
                "errors": sum(r.get("errors", 0) for r in all_results)
            }
            
            # Combine all processing times
            for r in all_results:
                aggregated["processing_times"].extend(r.get("processing_times", []))
                aggregated["loading_times"].extend(r.get("loading_times", []))
                aggregated["inference_times"].extend(r.get("inference_times", []))
            
            # Use last run's server stats
            if "server_stats" in all_results[-1]:
                aggregated["server_stats"] = all_results[-1]["server_stats"]
            
            print_results(server_name, aggregated, len(audio_files) * args.repeat)
        else:
            print_results(server_name, all_results[0], len(audio_files))
    
    # Compare results if both servers were tested
    if args.server == "both" and len(all_results) == 2:
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        
        if "error" not in all_results[0] and "error" not in all_results[1]:
            original_time = all_results[0]["total_time"]
            optimized_time = all_results[1]["total_time"]
            speedup = original_time / optimized_time
            
            print(f"Speedup: {speedup:.2f}x")
            print(f"Time saved: {original_time - optimized_time:.2f} seconds")
            print(f"Percentage improvement: {(1 - optimized_time/original_time) * 100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())