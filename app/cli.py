from __future__ import annotations

import argparse
import logging

from app.train import build_and_save


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Pocket Spellbook AI tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Load spells, preprocess and train the search model")
    build.add_argument("--source", choices=["auto", "api", "file"], default="auto")
    args = parser.parse_args()

    if args.command == "build":
        build_and_save(source=args.source)
        print("Model saved")


if __name__ == "__main__":
    main()
