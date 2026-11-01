"""Verify release file hashes and parse Python/JSON/TOML without importing the model."""
import ast
import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    checked = 0
    for name, expected in manifest["files"].items():
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Release asset missing: {name}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if path.stat().st_size != expected["bytes"] or actual != expected["sha256"]:
            raise ValueError(f"Release asset changed: {name}")
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path), feature_version=(3, 12))
        elif path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8-sig"))
        elif path.suffix == ".toml":
            tomllib.loads(path.read_text(encoding="utf-8-sig"))
        checked += 1
    print(f"Release integrity OK: {checked} files; Python/JSON/TOML parsed; no model or API calls.")

if __name__ == "__main__":
    main()
