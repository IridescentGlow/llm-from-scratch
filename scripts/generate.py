"""
Entry point for text generation from a checkpoint.
Usage: python scripts/generate.py --checkpoint checkpoints/latest.pt --prompt "..."
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    raise NotImplementedError("Implement after stage 3 (docs/03-architecture.md).")

if __name__ == "__main__":
    main()
