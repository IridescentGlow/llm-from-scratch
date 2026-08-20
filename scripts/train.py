"""
Entry point for pretraining. Thin — logic lives in src/llm_from_scratch/train/.
Usage: python scripts/train.py --config configs/small.yaml
"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise NotImplementedError("Implement after stage 4 (docs/04-pretraining.md).")

if __name__ == "__main__":
    main()
