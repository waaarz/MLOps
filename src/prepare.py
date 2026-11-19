from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "validation", "test")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    image_size = int(params["data"]["image_size"])

    raw_root = Path("data/raw")
    processed_root = Path("data/processed")
    manifests_root = processed_root / "manifests"
    reports_root = Path("reports")

    # PREPARE is intentionally separate from VALIDATE. In the DVC DAG, PREPARE
    # must depend on the successful data-quality report.
    if not raw_root.exists():
        raise FileNotFoundError("data/raw is missing")

    if processed_root.exists():
        shutil.rmtree(processed_root)
    processed_root.mkdir(parents=True)
    manifests_root.mkdir(parents=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    summary = {"image_size": image_size, "splits": {}, "classes": []}

    for split in SPLITS:
        class_dirs = sorted(p for p in (raw_root / split).iterdir() if p.is_dir())
        class_names = [p.name for p in class_dirs]
        if not summary["classes"]:
            summary["classes"] = class_names

        rows = []
        class_counts = {}
        for class_dir in class_dirs:
            out_class = processed_root / split / class_dir.name
            out_class.mkdir(parents=True, exist_ok=True)
            count = 0
            for src in sorted(class_dir.rglob("*")):
                if src.suffix.lower() not in IMAGE_EXTENSIONS or not src.is_file():
                    continue
                with Image.open(src) as img:
                    rgb = img.convert("RGB")
                    resized = rgb.resize((image_size, image_size), Image.Resampling.BILINEAR)
                    dst = out_class / f"{src.stem}.jpg"
                    resized.save(dst, "JPEG", quality=92)
                count += 1
                rows.append({
                    "relative_path": dst.as_posix(),
                    "label": class_dir.name,
                    "source_sha256": file_sha256(src),
                    "processed_sha256": file_sha256(dst),
                })
            class_counts[class_dir.name] = count

        manifest = pd.DataFrame(rows).sort_values(["label", "relative_path"])
        manifest.to_csv(manifests_root / f"{split}.csv", index=False, lineterminator="\n")
        summary["splits"][split] = {"images": int(len(manifest)), "class_counts": class_counts}

    (reports_root / "data_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
