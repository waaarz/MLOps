from __future__ import annotations

import json
from pathlib import Path
from PIL import Image

EXPECTED_SPLITS = {"train": 1034, "validation": 133, "test": 128}
EXPECTED_CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _list_images(class_dir: Path) -> list[Path]:
    """Sorted image files of one class folder (sorted => deterministic report)."""
    return sorted(
        p for p in class_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_dataset(raw_root: Path) -> dict:
    """Validate the raw dataset and return a deterministic quality report.

    Checks (any failure raises, so that the DVC stage stops):
      1. required splits exist;
      2. exactly the expected classes exist in every split;
      3. expected image counts are present;
      4. every image can be fully decoded by Pillow (`Image.load()`);
      5. no split or class is empty.

    The report contains no timestamp, absolute path or other volatile value,
    so the DVC output only changes when the data quality content changes.
    """
    raw_root = Path(raw_root)
    report_splits: dict[str, dict] = {}
    total = 0

    for split, expected_count in EXPECTED_SPLITS.items():
        split_dir = raw_root / split
        # 1. split exists
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        # 2. exactly the expected classes
        present = {p.name for p in split_dir.iterdir() if p.is_dir()}
        if present != set(EXPECTED_CLASSES):
            raise RuntimeError(
                f"Split '{split}' has classes {sorted(present)}, "
                f"expected {sorted(EXPECTED_CLASSES)}"
            )

        class_counts: dict[str, int] = {}
        for class_name in sorted(EXPECTED_CLASSES):
            images = _list_images(split_dir / class_name)
            # 5. no empty class
            if not images:
                raise RuntimeError(f"Class '{class_name}' is empty in split '{split}'")
            # 4. full pixel decode (Image.open only reads the header)
            for image_path in images:
                with Image.open(image_path) as img:
                    img.load()
            class_counts[class_name] = len(images)

        split_total = sum(class_counts.values())
        # 3. exact count
        if split_total != expected_count:
            raise RuntimeError(
                f"Split '{split}' has {split_total} images, expected {expected_count}"
            )
        report_splits[split] = {"images": split_total, "class_counts": class_counts}
        total += split_total

    return {
        "status": "PASS",
        "classes": sorted(EXPECTED_CLASSES),
        "splits": report_splits,
        "total_images": total,
    }


def main() -> None:
    report = validate_dataset(Path("data/raw"))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
