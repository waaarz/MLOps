from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

DATASET_URLS = {
    "train": [
        "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/train.zip?download=true",
        "https://storage.googleapis.com/ibeans/train.zip",
    ],
    "validation": [
        "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/validation.zip?download=true",
        "https://storage.googleapis.com/ibeans/validation.zip",
    ],
    "test": [
        "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data/test.zip?download=true",
        "https://storage.googleapis.com/ibeans/test.zip",
    ],
}
EXPECTED_SHA256 = {
    "train": "284fe8456ce20687f4367ae7ad94a64577e7f9fde2c2c6b1c74340ab5dc82715",
    "validation": "90b7aa1c26d91d9afff07a30bbc67a5ea34f1f1397f068d8675be09d7d0c602d",
    "test": "ca67b15d960d1e2fd9d23fd8498ce86818ead90755c630a43baf19ec4af09312",
}
EXPECTED_COUNTS = {"train": 1034, "validation": 133, "test": 128}
CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(url: str, destination: Path, retries: int = 3) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "AI53-MLOps-Lab/2.0"})
            with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as out:
                shutil.copyfileobj(response, out, length=1024 * 1024)
            return
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def download_with_fallback(urls: list[str], destination: Path) -> None:
    errors = []
    for url in urls:
        try:
            print(f"Trying {url}")
            download_one(url, destination)
            return
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("All public download sources failed:\n" + "\n".join(errors))


def count_images(split_dir: Path) -> int:
    return sum(
        1
        for path in split_dir.rglob("*")
        if path.suffix.lower() in EXTENSIONS and path.is_file()
    )


def validate_layout(raw_dir: Path) -> None:
    for split, expected in EXPECTED_COUNTS.items():
        split_dir = raw_dir / split
        if not split_dir.is_dir():
            raise RuntimeError(f"Missing split directory: {split_dir}")
        present_classes = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if present_classes != CLASSES:
            raise RuntimeError(f"Unexpected classes in {split}: {sorted(present_classes)}")
        actual = count_images(split_dir)
        if actual != expected:
            raise RuntimeError(f"Unexpected image count in {split}: {actual}; expected {expected}")
        print(f"{split:10s}: {actual:4d} images OK")


def safe_extract(archive: Path, raw_dir: Path) -> None:
    root = raw_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (raw_dir / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        zf.extractall(raw_dir)


def extract_archive(archive: Path, raw_dir: Path, split: str) -> None:
    safe_extract(archive, raw_dir)
    if not (raw_dir / split).is_dir():
        raise RuntimeError(f"Expected {raw_dir / split} after extracting {archive.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and verify the public iBean/Beans dataset")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--cache-dir", default=".dataset-cache")
    parser.add_argument(
        "--archive-dir", default=None,
        help="Offline mode: directory already containing train.zip, validation.zip and test.zip",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    cache_dir = Path(args.cache_dir)
    archive_dir = Path(args.archive_dir) if args.archive_dir else None

    if all((raw_dir / split).is_dir() for split in EXPECTED_COUNTS) and not args.force:
        print("Existing dataset found; validating layout instead of redownloading.")
        validate_layout(raw_dir)
        return

    if args.force and raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split, urls in DATASET_URLS.items():
        if archive_dir:
            archive = archive_dir / f"{split}.zip"
            if not archive.exists():
                raise FileNotFoundError(f"Missing offline archive: {archive}")
        else:
            archive = cache_dir / f"{split}.zip"
            if args.force or not archive.exists():
                print(f"Downloading {split}.zip ...")
                download_with_fallback(urls, archive)

        actual_hash = sha256(archive)
        if actual_hash != EXPECTED_SHA256[split]:
            raise RuntimeError(
                f"SHA-256 mismatch for {archive.name}: expected {EXPECTED_SHA256[split]}, got {actual_hash}"
            )
        print(f"{archive.name}: SHA-256 OK")
        extract_archive(archive, raw_dir, split)

    validate_layout(raw_dir)
    print("Dataset ready in data/raw/.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
