"""
Clear cached features and re-extract them to ensure they're complete.
This fixes issues where feature extraction was interrupted.
"""

import shutil
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description='Clear feature cache and re-extract')
    parser.add_argument('--cache_dir', type=str, default='.feature_cache',
                        help='Cache directory to clear')
    parser.add_argument('--confirm', action='store_true',
                        help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    
    if not cache_dir.exists():
        print(f"Cache directory {cache_dir} doesn't exist. Nothing to clear.")
        return
    
    print(f"Cache directory: {cache_dir}")
    print(f"Contents:")
    for f in cache_dir.iterdir():
        size = f.stat().st_size if f.is_file() else 0
        print(f"  {f.name} ({size:,} bytes)")
    
    if not args.confirm:
        response = input(f"\n⚠️  Delete all cached features? (yes/no): ")
        if response.lower() != 'yes':
            print("Cancelled.")
            return
    
    # Delete cache directory
    shutil.rmtree(cache_dir)
    print(f"\n✅ Deleted {cache_dir}")
    print("\nNow run training again to re-extract features:")
    print("  python train_on_wikibio.py --device cpu --test_size 0.3 --random_seed 42")
    print("\nThis will take time but ensures features are complete and correct.")

if __name__ == '__main__':
    main()
