"""
Entry point for evaluation (val loss/perplexity + sample generations).
Usage: python scripts/evaluate.py --checkpoint checkpoints/latest.pt
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    raise NotImplementedError("Implement after stage 5 (docs/05-evaluation.md).")

if __name__ == "__main__":
    main()
