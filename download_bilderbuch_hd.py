#!/usr/bin/env python3
"""M.O.L.O.C.H. HD Bilderbuch - Emotions + Gender + Age

Laedt hochaufgeloeste Gesichtsbilder fuer ALLE Kalibrierungsphasen:
  1. Emotionen (Happy, Sad, Angry, Neutral, Surprised) - 200 pro Klasse
  2. Gender (M, F) - 200 pro Klasse
  3. Age (kind, jung, mittel, alt) - 200 pro Klasse

Quellen:
  - UTKFace (GitHub): 23k+ Face Crops, 200x200, Age+Gender+Ethnicity im Dateinamen
  - AffectNet (HuggingFace): Emotion Labels, HD
  - DiffusionFER (HuggingFace): Synthetic, Emotion Labels

Alles auf SSD2: /mnt/moloch-data/reference/
"""
import os
import sys
import random
import subprocess
import zipfile
import shutil
from pathlib import Path

# ============================================================
# KONFIGURATION
# ============================================================
BASE_DIR = "/mnt/moloch-data/reference"
TMP_DIR = "/mnt/moloch-data/reference/_tmp_hd"
MAX_PER_CLASS = 200

EMOTIONS_HD = os.path.join(BASE_DIR, "emotions_hd")
GENDER_DIR = os.path.join(BASE_DIR, "gender")
AGE_DIR = os.path.join(BASE_DIR, "age")

EMOTION_CLASSES = ["happy", "sad", "angry", "neutral", "surprised"]
GENDER_CLASSES = ["male", "female"]
AGE_CLASSES = ["kind", "jung", "mittel", "alt"]

# UTKFace Filename: [age]_[gender]_[race]_[date].jpg
# Gender: 0=male, 1=female
# Race: 0=White, 1=Black, 2=Asian, 3=Indian, 4=Other

# Age Mapping fuer MOLOCH
AGE_MAP = {
    range(0, 13): "kind",     # 0-12
    range(13, 30): "jung",    # 13-29
    range(30, 55): "mittel",  # 30-54
    range(55, 120): "alt",    # 55+
}

print("=" * 50)
print("M.O.L.O.C.H. HD Bilderbuch")
print("=" * 50)

# Ordner erstellen
for d in [EMOTIONS_HD, GENDER_DIR, AGE_DIR, TMP_DIR]:
    os.makedirs(d, exist_ok=True)
for cls in EMOTION_CLASSES:
    os.makedirs(os.path.join(EMOTIONS_HD, cls), exist_ok=True)
for cls in GENDER_CLASSES:
    os.makedirs(os.path.join(GENDER_DIR, cls), exist_ok=True)
for cls in AGE_CLASSES:
    os.makedirs(os.path.join(AGE_DIR, cls), exist_ok=True)


def count_images(directory):
    """Zaehle Bilder in einem Ordner."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def needs_more(directory, target=MAX_PER_CLASS):
    """Braucht der Ordner noch Bilder?"""
    return count_images(directory) < target


# ============================================================
# QUELLE 1: UTKFace (Gender + Age + neutrale Emotion)
# ============================================================
print("\n--- Quelle 1: UTKFace (Gender + Age) ---")

# UTKFace Parts von GitHub
UTKFACE_URLS = [
    "https://huggingface.co/datasets/IQTesting/UTKFace/resolve/main/data/UTKFace.tar.gz",
]

# Alternative: Kaggle UTKFace (falls HF nicht klappt)
UTKFACE_KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/jangedoo/utkface-new"

utk_dir = os.path.join(TMP_DIR, "utkface")
utk_tar = os.path.join(TMP_DIR, "UTKFace.tar.gz")

any_gender_needed = needs_more(os.path.join(GENDER_DIR, "male")) or \
                    needs_more(os.path.join(GENDER_DIR, "female"))
any_age_needed = any(needs_more(os.path.join(AGE_DIR, c)) for c in AGE_CLASSES)

if any_gender_needed or any_age_needed:
    if not os.path.isdir(utk_dir) or len(os.listdir(utk_dir)) < 100:
        print("  Downloading UTKFace...")
        for url in UTKFACE_URLS:
            print(f"  URL: {url}")
            r = subprocess.run(
                ["wget", "-O", utk_tar, "--timeout=60", "--tries=3",
                 "--progress=dot:giga", url],
                timeout=600
            )
            if r.returncode == 0:
                break
        else:
            print("  HuggingFace fehlgeschlagen, versuche Alternative...")
            # Versuche direkten Download
            r = subprocess.run(
                ["wget", "-O", utk_tar, "--timeout=60", "--tries=3",
                 "--progress=dot:giga",
                 "https://drive.google.com/uc?export=download&id=0BxYys69jI14kYVM3aVhKS1VhRUk"],
                timeout=600
            )

        if os.path.exists(utk_tar):
            print("  Extrahiere...")
            os.makedirs(utk_dir, exist_ok=True)
            import tarfile
            try:
                with tarfile.open(utk_tar, 'r:gz') as tf:
                    tf.extractall(utk_dir)
                print("  Extraktion OK")
            except Exception as e:
                print(f"  tar.gz Fehler: {e}, versuche ZIP...")
                try:
                    with zipfile.ZipFile(utk_tar, 'r') as zf:
                        zf.extractall(utk_dir)
                    print("  ZIP Extraktion OK")
                except:
                    print("  Auch kein ZIP!")

    # UTKFace Bilder verarbeiten
    # Suche alle JPG Dateien (rekursiv)
    utk_images = []
    for root, dirs, files in os.walk(utk_dir):
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                utk_images.append(os.path.join(root, f))

    print(f"  {len(utk_images)} UTKFace Bilder gefunden")

    gender_count = {"male": 0, "female": 0}
    age_count = {c: 0 for c in AGE_CLASSES}
    neutral_count = 0

    random.seed(42)
    random.shuffle(utk_images)

    for img_path in utk_images:
        fname = os.path.basename(img_path)
        parts = fname.split("_")

        if len(parts) < 3:
            continue

        try:
            age = int(parts[0])
            gender_code = int(parts[1])
        except (ValueError, IndexError):
            continue

        gender = "male" if gender_code == 0 else "female"

        # Age Bucket
        age_class = None
        for age_range, cls in AGE_MAP.items():
            if age in age_range:
                age_class = cls
                break

        # Gender kopieren
        gender_dir = os.path.join(GENDER_DIR, gender)
        if needs_more(gender_dir):
            n = count_images(gender_dir)
            dst = os.path.join(gender_dir, f"utk_{n:04d}.jpg")
            shutil.copy2(img_path, dst)
            gender_count[gender] += 1

        # Age kopieren
        if age_class:
            age_dir_cls = os.path.join(AGE_DIR, age_class)
            if needs_more(age_dir_cls):
                n = count_images(age_dir_cls)
                dst = os.path.join(age_dir_cls, f"utk_{n:04d}.jpg")
                shutil.copy2(img_path, dst)
                age_count[age_class] += 1

        # Neutral-Emotion (UTKFace hat keine Emotion-Labels, aber neutrale Ausdruecke)
        emo_neutral = os.path.join(EMOTIONS_HD, "neutral")
        if needs_more(emo_neutral) and neutral_count < MAX_PER_CLASS:
            n = count_images(emo_neutral)
            dst = os.path.join(emo_neutral, f"utk_{n:04d}.jpg")
            shutil.copy2(img_path, dst)
            neutral_count += 1

    print(f"  Gender: {gender_count}")
    print(f"  Age: {age_count}")
    print(f"  Neutral Emotions: {neutral_count}")
else:
    print("  Gender + Age bereits voll, ueberspringe")


# ============================================================
# QUELLE 2: HuggingFace Emotion Datasets
# ============================================================
print("\n--- Quelle 2: HuggingFace Emotions ---")

any_emo_needed = any(needs_more(os.path.join(EMOTIONS_HD, c)) for c in EMOTION_CLASSES)

if any_emo_needed:
    try:
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

        # Versuche verschiedene Repos
        repos = [
            ("FER-Universe/DiffusionFER", {}),
            ("Mauregato/affectnet_short", {}),
            ("ddPn08/facial-expression-recognition", {}),
        ]

        for repo_name, kwargs in repos:
            if not any(needs_more(os.path.join(EMOTIONS_HD, c))
                       for c in EMOTION_CLASSES if c != "neutral"):
                print("  Alle Emotions-Klassen voll!")
                break

            print(f"\n  Versuche: {repo_name}")
            try:
                ds = load_dataset(repo_name, split="train", **kwargs)
                print(f"  Geladen: {len(ds)} Bilder")
                cols = ds.column_names
                print(f"  Spalten: {cols}")

                # Label + Image Spalte finden
                label_col = next((c for c in ["label", "emotion", "expression"]
                                  if c in cols), None)
                img_col = next((c for c in ["image", "img", "pixel_values"]
                                if c in cols), None)

                if not label_col or not img_col:
                    print(f"  Spalten nicht erkannt")
                    continue

                # Label-Namen
                label_names = None
                if hasattr(ds.features.get(label_col, None), 'names'):
                    label_names = ds.features[label_col].names
                    print(f"  Label-Namen: {label_names}")

                saved = 0
                indices = list(range(len(ds)))
                random.seed(42)
                random.shuffle(indices)

                for i in indices:
                    if not any(needs_more(os.path.join(EMOTIONS_HD, c))
                               for c in EMOTION_CLASSES):
                        break

                    row = ds[i]
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
                    if isinstance(img, Image.Image):
                        if img.width >= 64 and img.height >= 64:
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

                print(f"  Gespeichert: {saved} Bilder")

            except Exception as e:
                print(f"  Fehler: {e}")

    except ImportError:
        print("  'datasets' library nicht installiert!")
else:
    print("  Emotions bereits voll, ueberspringe")


# ============================================================
# AUFRAEUMEN
# ============================================================
if os.path.exists(utk_tar):
    os.remove(utk_tar)
    print(f"\nTemp-Datei geloescht: {utk_tar}")


# ============================================================
# REPORT
# ============================================================
print("\n" + "=" * 50)
print("M.O.L.O.C.H. HD BILDERBUCH - REPORT")
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
