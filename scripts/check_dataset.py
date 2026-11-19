from pathlib import Path

EXPECTED = {"train": 1034, "validation": 133, "test": 128}
CLASSES = {"angular_leaf_spot", "bean_rust", "healthy"}
EXTENSIONS = {".jpg", ".jpeg", ".png"}

raw = Path("data/raw")
all_ok = True
for split, expected_total in EXPECTED.items():
    split_dir = raw / split
    print(f"\n[{split}]")
    if not split_dir.is_dir():
        print("  MISSING SPLIT")
        all_ok = False
        continue
    present = {p.name for p in split_dir.iterdir() if p.is_dir()}
    if present != CLASSES:
        print(f"  CLASS SET MISMATCH: {sorted(present)}")
        all_ok = False
    total = 0
    for class_name in sorted(CLASSES):
        class_dir = split_dir / class_name
        count = sum(
            1
            for path in class_dir.rglob("*")
            if path.suffix.lower() in EXTENSIONS and path.is_file()
        )
        total += count
        print(f"  {class_name:20s} {count:4d}")
    print(f"  {'TOTAL':20s} {total:4d} (expected {expected_total})")
    all_ok &= total == expected_total
raise SystemExit(0 if all_ok else 1)
