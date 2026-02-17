#!/usr/bin/env python3
"""M.O.L.O.C.H. HD Emotions Bilderbuch

Laedt hochaufgeloeste Gesichtsbilder mit Emotion-Labels aus mehreren
HuggingFace Quellen und organisiert sie passend fuer MOLOCHs Pipeline:
  SCRFD Face Detection (NPU) -> Face Crop -> FER+ Emotion (CPU)

Quellen:
  1. DiffusionFER (FER-Universe) - 512px Synthetic, CC0, 7 Klassen
  2. AffectNet Subsets auf HuggingFace (falls verfuegbar)

Ziel: /mnt/moloch-data/reference/emotions_hd/{happy,sad,angry,neutral,surprised}/
      je ~200 hochaufgeloeste Bilder pro Kategorie
"""
import os
import sys
import random
import shutil
from pathlib import Path
from PIL import Image

DEST = "/mnt/moloch-data/reference/emotions_hd"
MAX_PER_CLASS = 200

# MOLOCH's FER+ 5 Klassen
TARGET_CLASSES = ["happy", "sad", "angry", "neutral", "surprised"]

# Mapping verschiedener Quellen -> MOLOCHs 5 Klassen
EMOTION_REMAP = {
    # Standard 7-class
    "happy": "happy", "happiness": "happy", "joy": "happy",
    "sad": "sad", "sadness": "sad",
    "angry": "angry", "anger": "angry",
    "neutral": "neutral",
    "surprise": "surprised", "surprised": "surprised",
    "fear": "surprised",    # Aehnliche Gesichtszuege
    "disgust": "angry",     # Aehnliche Gesichtszuege
    "contempt": "angry",
    # LAION EmoNet Fine-grained -> Basic
    "amusement": "happy", "elation": "happy", "contentment": "happy",
    "affection": "happy", "pride": "happy", "triumph": "happy",
    "excitement": "surprised", "astonishment": "surprised",
    "concentration": "neutral", "confusion": "neutral",
    "fatigue": "sad", "pain": "sad", "shame": "sad",
    "guilt": "sad", "disappointment": "sad", "grief": "sad",
    "anxiety": "surprised", "distress": "sad",
    "irritation": "angry", "frustration": "angry",
    "boredom": "neutral", "calm": "neutral",
}

print("=== M.O.L.O.C.H. HD Emotions Bilderbuch ===")
print(f"Ziel: {DEST}")
print(f"Klassen: {TARGET_CLASSES}")
print(f"Max pro Klasse: {MAX_PER_CLASS}")

os.makedirs(DEST, exist_ok=True)
for cls in TARGET_CLASSES:
    os.makedirs(os.path.join(DEST, cls), exist_ok=True)

counts = {cls: 0 for cls in TARGET_CLASSES}

# Zaehle bestehende Bilder
for cls in TARGET_CLASSES:
    d = os.path.join(DEST, cls)
    existing = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    counts[cls] = len(existing)
    if counts[cls] > 0:
        print(f"  {cls}: {counts[cls]} Bilder vorhanden")


def save_image(img, cls, prefix, idx):
    """Bild speichern wenn Platz in der Klasse."""
    if counts[cls] >= MAX_PER_CLASS:
        return False

    # Qualitaetscheck: Mindestens 64x64
    if img.width < 64 or img.height < 64:
        return False

    # Als JPG speichern (platzsparend)
    fname = f"{prefix}_{idx:05d}.jpg"
    fpath = os.path.join(DEST, cls, fname)

    # RGB konvertieren (falls RGBA oder Grayscale)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img.save(fpath, "JPEG", quality=90)
    counts[cls] += 1
    return True


# ============================================================
# QUELLE 1: DiffusionFER (FER-Universe) - Synthetic, CC0
# ============================================================
print("\n--- Quelle 1: DiffusionFER ---")
try:
    from datasets import load_dataset

    ds = load_dataset("FER-Universe/DiffusionFER", split="train", trust_remote_code=True)
    print(f"  Geladen: {len(ds)} Bilder")

    # Spalten checken
    cols = ds.column_names
    print(f"  Spalten: {cols}")

    # Emotion-Spalte finden
    label_col = None
    for c in ["label", "emotion", "expression", "category"]:
        if c in cols:
            label_col = c
            break

    img_col = None
    for c in ["image", "img", "pixel_values"]:
        if c in cols:
            img_col = c
            break

    if label_col and img_col:
        saved = 0
        for i, row in enumerate(ds):
            label = str(row[label_col]).lower().strip()
            target = EMOTION_REMAP.get(label)
            if not target:
                continue
            if counts[target] >= MAX_PER_CLASS:
                continue

            img = row[img_col]
            if isinstance(img, Image.Image):
                if save_image(img, target, "difffer", i):
                    saved += 1

        print(f"  Gespeichert: {saved} Bilder")
    else:
        print(f"  Label/Image Spalte nicht gefunden: label={label_col}, img={img_col}")
        # Zeige erste Zeile zum Debuggen
        print(f"  Erste Zeile: {ds[0]}")

except Exception as e:
    print(f"  Fehler: {e}")


# ============================================================
# QUELLE 2: AffectNet Subsets auf HuggingFace
# ============================================================
print("\n--- Quelle 2: AffectNet (HuggingFace Mirrors) ---")

affectnet_repos = [
    "Mauregato/affectnet_short",
    "ddPn08/facial-expression-recognition",
]

for repo in affectnet_repos:
    if all(counts[c] >= MAX_PER_CLASS for c in TARGET_CLASSES):
        print("  Alle Klassen voll!")
        break

    print(f"\n  Versuche: {repo}")
    try:
        ds = load_dataset(repo, split="train", trust_remote_code=True)
        print(f"  Geladen: {len(ds)} Bilder")
        cols = ds.column_names
        print(f"  Spalten: {cols}")

        # Label-Spalte finden
        label_col = None
        for c in ["label", "emotion", "expression", "category", "labels"]:
            if c in cols:
                label_col = c
                break

        img_col = None
        for c in ["image", "img", "pixel_values", "face"]:
            if c in cols:
                img_col = c
                break

        if not label_col or not img_col:
            print(f"  Spalten nicht erkannt, ueberspringe")
            continue

        # Label-Mapping aufbauen (kann numerisch sein)
        # Versuche features zu lesen
        label_names = None
        if hasattr(ds.features[label_col], 'names'):
            label_names = ds.features[label_col].names
            print(f"  Label-Namen: {label_names}")

        saved = 0
        indices = list(range(len(ds)))
        random.seed(42)
        random.shuffle(indices)

        for i in indices:
            if all(counts[c] >= MAX_PER_CLASS for c in TARGET_CLASSES):
                break

            row = ds[i]
            raw_label = row[label_col]

            # Numerisch -> String
            if isinstance(raw_label, int) and label_names:
                label = label_names[raw_label].lower()
            else:
                label = str(raw_label).lower().strip()

            target = EMOTION_REMAP.get(label)
            if not target or counts[target] >= MAX_PER_CLASS:
                continue

            img = row[img_col]
            if isinstance(img, Image.Image):
                if save_image(img, target, repo.split("/")[-1], i):
                    saved += 1

        print(f"  Gespeichert: {saved} Bilder")

    except Exception as e:
        print(f"  Fehler: {e}")


# ============================================================
# QUELLE 3: LAION EmoNet-Face (Synthetic, CC-BY 4.0)
# ============================================================
print("\n--- Quelle 3: LAION EmoNet-Face ---")

if not all(counts[c] >= MAX_PER_CLASS for c in TARGET_CLASSES):
    try:
        # Versuche das kleinere HQ subset zuerst
        ds = load_dataset("laion/emonet-face-big", split="train",
                         trust_remote_code=True, streaming=True)
        print("  Streaming gestartet...")

        saved = 0
        for i, row in enumerate(ds):
            if all(counts[c] >= MAX_PER_CLASS for c in TARGET_CLASSES):
                print(f"  Alle Klassen voll nach {i} Bildern!")
                break

            # Max 50000 Bilder scannen
            if i > 50000:
                print(f"  50000 Bilder gescannt, stoppe")
                break

            cols = list(row.keys())

            # Label finden
            label_col = None
            for c in ["label", "emotion", "expression", "category"]:
                if c in cols:
                    label_col = c
                    break

            if not label_col:
                if i == 0:
                    print(f"  Spalten: {cols}")
                    print(f"  Erste Zeile: {row}")
                continue

            label = str(row[label_col]).lower().strip()
            target = EMOTION_REMAP.get(label)
            if not target or counts[target] >= MAX_PER_CLASS:
                continue

            img_col = None
            for c in ["image", "img"]:
                if c in cols:
                    img_col = c
                    break

            if img_col and isinstance(row[img_col], Image.Image):
                if save_image(row[img_col], target, "emonet", i):
                    saved += 1
                    if saved % 50 == 0:
                        print(f"  ... {saved} Bilder gespeichert "
                              f"({dict((k,v) for k,v in counts.items())})")

        print(f"  Gespeichert: {saved} Bilder")

    except Exception as e:
        print(f"  Fehler: {e}")
else:
    print("  Alle Klassen bereits voll, ueberspringe")


# ============================================================
# Report
# ============================================================
print("\n=== M.O.L.O.C.H. HD EMOTIONS BILDERBUCH ===")
total = 0
for cls in TARGET_CLASSES:
    d = os.path.join(DEST, cls)
    n = len([f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total += n

    # Durchschnittliche Aufloesung der ersten 5 Bilder
    files = sorted(os.listdir(d))[:5]
    resolutions = []
    for f in files:
        try:
            img = Image.open(os.path.join(d, f))
            resolutions.append(f"{img.width}x{img.height}")
        except:
            pass
    res_str = ", ".join(resolutions[:3]) if resolutions else "?"

    status = "OK" if n >= MAX_PER_CLASS else f"NUR {n}"
    print(f"  {cls:12s}: {n:4d} Bilder  ({res_str})  [{status}]")

print(f"  {'GESAMT':12s}: {total:4d} Bilder")
print(f"\nZiel: {DEST}")

if total < len(TARGET_CLASSES) * 50:
    print("\nWARNUNG: Weniger als 50 Bilder pro Klasse!")
    print("Manuell pruefen welche Quellen funktioniert haben.")
