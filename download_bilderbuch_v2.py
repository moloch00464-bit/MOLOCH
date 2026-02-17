#!/usr/bin/env python3
"""M.O.L.O.C.H. HD Bilderbuch v2 - via HuggingFace datasets Streaming.

Laedt Referenzbilder per Streaming (kein voller Download noetig):
  1. UTKFace -> Gender (M/F) + Age (kind/jung/mittel/alt) + Neutral Emotions
  2. FER+ Datasets -> Emotion HD Bilder (Happy, Sad, Angry, Neutral, Surprised)

Alles auf SSD2: /mnt/moloch-data/reference/
"""
import os
import sys
import random
import shutil
from pathlib import Path

# ============================================================
# KONFIGURATION
# ============================================================
BASE_DIR = "/mnt/moloch-data/reference"
MAX_PER_CLASS = 200

EMOTIONS_HD = os.path.join(BASE_DIR, "emotions_hd")
GENDER_DIR = os.path.join(BASE_DIR, "gender")
AGE_DIR = os.path.join(BASE_DIR, "age")

EMOTION_CLASSES = ["happy", "sad", "angry", "neutral", "surprised"]
GENDER_CLASSES = ["male", "female"]
AGE_CLASSES = ["kind", "jung", "mittel", "alt"]

# UTKFace: age_gender_race_date.jpg
# Gender: 0=male, 1=female
AGE_RANGES = {
    "kind": (0, 12),
    "jung": (13, 29),
    "mittel": (30, 54),
    "alt": (55, 120),
}


def count_images(directory):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def needs_more(directory, target=MAX_PER_CLASS):
    return count_images(directory) < target


# Ordner erstellen
for d in [EMOTIONS_HD, GENDER_DIR, AGE_DIR]:
    os.makedirs(d, exist_ok=True)
for cls in EMOTION_CLASSES:
    os.makedirs(os.path.join(EMOTIONS_HD, cls), exist_ok=True)
for cls in GENDER_CLASSES:
    os.makedirs(os.path.join(GENDER_DIR, cls), exist_ok=True)
for cls in AGE_CLASSES:
    os.makedirs(os.path.join(AGE_DIR, cls), exist_ok=True)


print("=" * 50)
print("M.O.L.O.C.H. HD Bilderbuch v2")
print("=" * 50)


# ============================================================
# QUELLE 1: UTKFace via HuggingFace datasets (Gender + Age)
# ============================================================
print("\n--- Quelle 1: UTKFace (Gender + Age + Neutral) ---")

any_gender_needed = needs_more(os.path.join(GENDER_DIR, "male")) or \
                    needs_more(os.path.join(GENDER_DIR, "female"))
any_age_needed = any(needs_more(os.path.join(AGE_DIR, c)) for c in AGE_CLASSES)

if any_gender_needed or any_age_needed:
    from datasets import load_dataset
    from PIL import Image

    # Verschiedene UTKFace Repos versuchen
    utk_repos = [
        "prdwb/UTKFace",
        "IQTesting/UTKFace",
        "Vasanthntr/UTKFace",
    ]

    utk_ds = None
    for repo in utk_repos:
        print(f"  Versuche: {repo}")
        try:
            utk_ds = load_dataset(repo, split="train", streaming=True)
            # Teste ob wir iterieren koennen
            sample = next(iter(utk_ds))
            cols = list(sample.keys())
            print(f"  OK! Spalten: {cols}")
            break
        except Exception as e:
            print(f"  Fehler: {e}")
            utk_ds = None

    if utk_ds is not None:
        gender_count = {"male": 0, "female": 0}
        age_count = {c: 0 for c in AGE_CLASSES}
        neutral_count = 0
        total_processed = 0

        # Checke Spalten-Layout
        sample_keys = list(sample.keys())
        has_age_col = "age" in sample_keys
        has_gender_col = "gender" in sample_keys
        has_filename = "file" in sample_keys or "filename" in sample_keys

        print(f"  Schema: age_col={has_age_col}, gender_col={has_gender_col}")

        for row in utk_ds:
            # Alles voll? Aufhoeren
            all_gender_full = not any(needs_more(os.path.join(GENDER_DIR, c))
                                      for c in GENDER_CLASSES)
            all_age_full = not any(needs_more(os.path.join(AGE_DIR, c))
                                   for c in AGE_CLASSES)
            neutral_full = not needs_more(os.path.join(EMOTIONS_HD, "neutral"))

            if all_gender_full and all_age_full and neutral_full:
                print("  Alle Klassen voll!")
                break

            total_processed += 1

            # Age + Gender extrahieren
            if has_age_col and has_gender_col:
                age = int(row["age"])
                gender_code = int(row["gender"])
            elif has_filename:
                fname = row.get("file", row.get("filename", ""))
                parts = os.path.basename(fname).split("_")
                if len(parts) < 3:
                    continue
                try:
                    age = int(parts[0])
                    gender_code = int(parts[1])
                except (ValueError, IndexError):
                    continue
            else:
                # Versuche aus dem Bildnamen
                continue

            gender = "male" if gender_code == 0 else "female"

            # Bild holen
            img = None
            for img_key in ["image", "img", "pixel_values"]:
                if img_key in row:
                    img = row[img_key]
                    break
            if img is None or not isinstance(img, Image.Image):
                continue

            if img.mode != "RGB":
                img = img.convert("RGB")

            # Min-Groesse
            if img.width < 64 or img.height < 64:
                continue

            # Gender speichern
            gender_dir = os.path.join(GENDER_DIR, gender)
            if needs_more(gender_dir):
                n = count_images(gender_dir)
                dst = os.path.join(gender_dir, f"utk_{n:04d}.jpg")
                img.save(dst, "JPEG", quality=90)
                gender_count[gender] += 1

            # Age speichern
            age_class = None
            for cls, (lo, hi) in AGE_RANGES.items():
                if lo <= age <= hi:
                    age_class = cls
                    break

            if age_class:
                age_dir = os.path.join(AGE_DIR, age_class)
                if needs_more(age_dir):
                    n = count_images(age_dir)
                    dst = os.path.join(age_dir, f"utk_{n:04d}.jpg")
                    img.save(dst, "JPEG", quality=90)
                    age_count[age_class] += 1

            # Neutral-Emotion
            emo_neutral = os.path.join(EMOTIONS_HD, "neutral")
            if needs_more(emo_neutral) and neutral_count < MAX_PER_CLASS:
                n = count_images(emo_neutral)
                dst = os.path.join(emo_neutral, f"utk_{n:04d}.jpg")
                img.save(dst, "JPEG", quality=90)
                neutral_count += 1

            if total_processed % 200 == 0:
                print(f"  ... {total_processed} verarbeitet: "
                      f"Gender={gender_count}, Age={age_count}, Neutral={neutral_count}")

        print(f"  FERTIG: {total_processed} verarbeitet")
        print(f"  Gender: {gender_count}")
        print(f"  Age: {age_count}")
        print(f"  Neutral Emotions: {neutral_count}")
    else:
        print("  KEIN UTKFace Repo erreichbar!")
else:
    print("  Gender + Age bereits voll, ueberspringe")


# ============================================================
# QUELLE 2: HuggingFace Emotion Datasets (HD)
# ============================================================
print("\n--- Quelle 2: HuggingFace Emotions (HD) ---")

any_emo_needed = any(needs_more(os.path.join(EMOTIONS_HD, c))
                     for c in EMOTION_CLASSES if c != "neutral")

if any_emo_needed:
    from datasets import load_dataset
    from PIL import Image

    EMOTION_REMAP = {
        "happy": "happy", "happiness": "happy", "joy": "happy",
        "sad": "sad", "sadness": "sad",
        "angry": "angry", "anger": "angry",
        "neutral": "neutral",
        "surprise": "surprised", "surprised": "surprised",
        "fear": "surprised",
        "disgust": "angry",
        "contempt": "angry",
    }

    # Standard FER Repos (per Streaming)
    repos = [
        ("ddPn08/facial-expression-recognition", {}),
        ("Mauregato/affectnet_short", {}),
    ]

    for repo_name, kwargs in repos:
        if not any(needs_more(os.path.join(EMOTIONS_HD, c))
                   for c in EMOTION_CLASSES if c != "neutral"):
            print("  Alle Emotion-Klassen voll!")
            break

        print(f"\n  Versuche: {repo_name}")
        try:
            ds = load_dataset(repo_name, split="train", streaming=True, **kwargs)

            # Erste Row checken
            first = next(iter(ds))
            cols = list(first.keys())
            print(f"  Spalten: {cols}")

            # Label + Image Spalte finden
            label_col = next((c for c in ["label", "emotion", "expression"]
                              if c in cols), None)
            img_col = next((c for c in ["image", "img", "pixel_values"]
                            if c in cols), None)

            if not label_col or not img_col:
                print(f"  Spalten nicht erkannt: label={label_col}, img={img_col}")
                continue

            # Label-Namen (aus Features)
            label_names = None
            try:
                info = load_dataset(repo_name, streaming=True)
                features = info["train"].features
                if hasattr(features.get(label_col, None), 'names'):
                    label_names = features[label_col].names
                    print(f"  Label-Namen: {label_names}")
            except:
                pass

            saved = 0
            for row in ds:
                if not any(needs_more(os.path.join(EMOTIONS_HD, c))
                           for c in EMOTION_CLASSES):
                    break

                raw_label = row[label_col]
                if isinstance(raw_label, int) and label_names:
                    label = label_names[raw_label].lower()
                else:
                    label = str(raw_label).lower().strip()

                target = EMOTION_REMAP.get(label)
                if not target:
                    continue

                emo_dir = os.path.join(EMOTIONS_HD, target)
                if not needs_more(emo_dir):
                    continue

                img = row[img_col]
                if not isinstance(img, Image.Image):
                    continue

                # Nur HD-Bilder (mindestens 64x64)
                if img.width < 64 or img.height < 64:
                    continue

                if img.mode != "RGB":
                    img = img.convert("RGB")

                n = count_images(emo_dir)
                prefix = repo_name.split("/")[-1][:8]
                fpath = os.path.join(emo_dir, f"{prefix}_{n:04d}.jpg")
                img.save(fpath, "JPEG", quality=90)
                saved += 1

                if saved % 100 == 0:
                    status = {c: count_images(os.path.join(EMOTIONS_HD, c))
                              for c in EMOTION_CLASSES}
                    print(f"  ... {saved} gespeichert: {status}")

            print(f"  Gespeichert: {saved} Bilder von {repo_name}")

        except Exception as e:
            print(f"  Fehler: {e}")
            import traceback
            traceback.print_exc()
else:
    print("  Emotions bereits voll, ueberspringe")


# ============================================================
# REPORT
# ============================================================
print("\n" + "=" * 50)
print("M.O.L.O.C.H. HD BILDERBUCH v2 - REPORT")
print("=" * 50)

sections = [
    ("EMOTIONEN (HD)", EMOTIONS_HD, EMOTION_CLASSES),
    ("GENDER", GENDER_DIR, GENDER_CLASSES),
    ("ALTER", AGE_DIR, AGE_CLASSES),
]

grand_total = 0
for title, base, classes in sections:
    print(f"\n  {title}:")
    section_total = 0
    for cls in classes:
        d = os.path.join(base, cls)
        n = count_images(d)
        section_total += n
        status = "OK" if n >= MAX_PER_CLASS else f"FEHLT {MAX_PER_CLASS - n}"
        print(f"    {cls:12s}: {n:4d} / {MAX_PER_CLASS}  [{status}]")
    grand_total += section_total
    print(f"    {'Summe':12s}: {section_total}")

print(f"\n  GESAMT: {grand_total} Bilder")
print(f"  Speicherort: {BASE_DIR}")
