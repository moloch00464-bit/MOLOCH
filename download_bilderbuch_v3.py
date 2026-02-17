#!/usr/bin/env python3
"""M.O.L.O.C.H. HD Bilderbuch v3 - FairFace (Gender + Age).

FairFace Dataset (HuggingFaceM4/FairFace):
  - 224x224 Face Crops
  - Gender: Male/Female
  - Age: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+

Mapping auf MOLOCHs 4 Altersklassen:
  - kind:   0-2, 3-9, 10-19 (0-19)
  - jung:   20-29 (20-29)
  - mittel: 30-39, 40-49 (30-49)
  - alt:    50-59, 60-69, 70+ (50+)

Alles auf SSD2: /mnt/moloch-data/reference/
"""
import os
import sys

BASE_DIR = "/mnt/moloch-data/reference"
MAX_PER_CLASS = 200
GENDER_DIR = os.path.join(BASE_DIR, "gender")
AGE_DIR = os.path.join(BASE_DIR, "age")

GENDER_CLASSES = ["male", "female"]
AGE_CLASSES = ["kind", "jung", "mittel", "alt"]

# FairFace Age Index -> MOLOCH Klasse
AGE_INDEX_MAP = {
    0: "kind",    # 0-2
    1: "kind",    # 3-9
    2: "kind",    # 10-19
    3: "jung",    # 20-29
    4: "mittel",  # 30-39
    5: "mittel",  # 40-49
    6: "alt",     # 50-59
    7: "alt",     # 60-69
    8: "alt",     # more than 70
}

# Gender Index -> Name
GENDER_MAP = {0: "male", 1: "female"}


def count_images(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def needs_more(directory, target=MAX_PER_CLASS):
    return count_images(directory) < target


# Ordner erstellen
for cls in GENDER_CLASSES:
    os.makedirs(os.path.join(GENDER_DIR, cls), exist_ok=True)
for cls in AGE_CLASSES:
    os.makedirs(os.path.join(AGE_DIR, cls), exist_ok=True)

print("=" * 50)
print("M.O.L.O.C.H. HD Bilderbuch v3 - FairFace")
print("=" * 50)

# Status vorher
print("\nAktueller Stand:")
for cls in GENDER_CLASSES:
    d = os.path.join(GENDER_DIR, cls)
    print(f"  Gender/{cls}: {count_images(d)}/{MAX_PER_CLASS}")
for cls in AGE_CLASSES:
    d = os.path.join(AGE_DIR, cls)
    print(f"  Age/{cls}: {count_images(d)}/{MAX_PER_CLASS}")

any_needed = any(needs_more(os.path.join(GENDER_DIR, c)) for c in GENDER_CLASSES) or \
             any(needs_more(os.path.join(AGE_DIR, c)) for c in AGE_CLASSES)

if not any_needed:
    print("\nAlles voll!")
    sys.exit(0)

print("\nLade FairFace Dataset (Streaming)...")
from datasets import load_dataset
from PIL import Image

ds = load_dataset("HuggingFaceM4/FairFace", "0.25", split="train", streaming=True)

gender_count = {"male": 0, "female": 0}
age_count = {c: 0 for c in AGE_CLASSES}
total = 0

for row in ds:
    # Alles voll?
    all_gender_full = not any(needs_more(os.path.join(GENDER_DIR, c)) for c in GENDER_CLASSES)
    all_age_full = not any(needs_more(os.path.join(AGE_DIR, c)) for c in AGE_CLASSES)
    if all_gender_full and all_age_full:
        print("\nAlle Klassen voll!")
        break

    total += 1
    gender_idx = row["gender"]
    age_idx = row["age"]
    img = row["image"]

    if not isinstance(img, Image.Image):
        continue
    if img.mode != "RGB":
        img = img.convert("RGB")

    gender = GENDER_MAP.get(gender_idx)
    age_class = AGE_INDEX_MAP.get(age_idx)

    # Gender speichern
    if gender:
        gdir = os.path.join(GENDER_DIR, gender)
        if needs_more(gdir):
            n = count_images(gdir)
            img.save(os.path.join(gdir, f"ff_{n:04d}.jpg"), "JPEG", quality=92)
            gender_count[gender] += 1

    # Age speichern
    if age_class:
        adir = os.path.join(AGE_DIR, age_class)
        if needs_more(adir):
            n = count_images(adir)
            img.save(os.path.join(adir, f"ff_{n:04d}.jpg"), "JPEG", quality=92)
            age_count[age_class] += 1

    if total % 200 == 0:
        print(f"  {total} verarbeitet: Gender={gender_count}, Age={age_count}")

print(f"\nFERTIG: {total} verarbeitet")
print(f"Gender: {gender_count}")
print(f"Age: {age_count}")

# Report
print("\n" + "=" * 50)
print("REPORT")
print("=" * 50)
for cls in GENDER_CLASSES:
    d = os.path.join(GENDER_DIR, cls)
    n = count_images(d)
    s = "OK" if n >= MAX_PER_CLASS else f"FEHLT {MAX_PER_CLASS - n}"
    print(f"  Gender/{cls:8s}: {n:4d}/{MAX_PER_CLASS} [{s}]")
for cls in AGE_CLASSES:
    d = os.path.join(AGE_DIR, cls)
    n = count_images(d)
    s = "OK" if n >= MAX_PER_CLASS else f"FEHLT {MAX_PER_CLASS - n}"
    print(f"  Age/{cls:8s}: {n:4d}/{MAX_PER_CLASS} [{s}]")
