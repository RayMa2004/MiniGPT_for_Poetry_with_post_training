"""Explicit phase dispatcher; help/offline phases do not import torch."""
import argparse
import importlib
import sys


def main(argv=None):
    argv=sys.argv[1:] if argv is None else list(argv)
    parser=argparse.ArgumentParser(description="MiniGPT v3: isolated AI-feedback post-training")
    parser.add_argument("phase",choices=("data","sample","judge","sft","reward","ppo","compare","pipeline","bestof"))
    if not argv or argv[0].startswith("-"):
        parser.parse_args(argv)
        return 0
    args=parser.parse_args(argv[:1])
    module=importlib.import_module("post_training."+args.phase)
    return module.main(argv[1:])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        print(f"post_training: {error}", file=sys.stderr)
        raise SystemExit(2)
