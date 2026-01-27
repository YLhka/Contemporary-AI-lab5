"""
Quick EDA for multimodal dataset (using tag instead of label).

Directory structure:
project5/
  └─ project5/
      ├─ train.txt              # guid,tag
      ├─ test_without_label.txt # optional
      ├─ data/
      │   ├─ {guid}.jpg / png ...
      │   └─ {guid}.txt         # text
      └─ eda_quick.py
"""

import re
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# optional jieba
try:
    import jieba
    HAVE_JIEBA = True
except Exception:
    HAVE_JIEBA = False


# ================= CONFIG =================
BASE_DIR = Path(__file__).resolve().parent
TRAIN_META = BASE_DIR / "train.txt"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "eda_output"
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

CJK_RE = re.compile(r'[\u4e00-\u9fff]')
HASHTAG_RE = re.compile(r"#\w+")
EMOJI_RE = re.compile(
    "[" 
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U00002700-\U000027BF"
    "\U0001F1E0-\U0001F1FF"
    "]+",
    flags=re.UNICODE
)
# ==========================================


def read_train_meta(path: Path) -> pd.DataFrame:
    """Read train.txt with columns (guid, tag)"""
    try:
        df = pd.read_csv(path, engine="python", dtype=str)
    except Exception:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = re.split(r"[,\s]+", line.strip())
                if len(parts) >= 2:
                    rows.append((parts[0], parts[1]))
        return pd.DataFrame(rows, columns=["guid", "tag"])

    cols = [c.lower().strip() for c in df.columns]

    guid_col = df.columns[cols.index("guid")] if "guid" in cols else df.columns[0]
    tag_col = df.columns[cols.index("tag")] if "tag" in cols else df.columns[1]

    out = pd.DataFrame()
    out["guid"] = df[guid_col].astype(str).str.strip()
    out["tag"] = df[tag_col].astype(str).str.strip()
    return out


def tokenize(text: str):
    if not text:
        return []
    if CJK_RE.search(text) and HAVE_JIEBA:
        return [t for t in jieba.lcut(text) if t.strip()]
    tokens = re.findall(r"[A-Za-z0-9']{2,}|#\w+|@\w+", text)
    return [t.lower() for t in tokens]


def extract_emojis(text: str):
    return EMOJI_RE.findall(text) if text else []


def load_text(guid: str) -> str:
    p = DATA_DIR / f"{guid}.txt"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            return ""
    return ""


def find_image(guid: str):
    for ext in IMAGE_EXTS:
        p = DATA_DIR / f"{guid}{ext}"
        if p.exists():
            return p
    return None


def eda(meta: pd.DataFrame):
    rows = []
    missing_images = []
    corrupted_images = []
    image_sizes = []

    for guid, tag in tqdm(meta.itertuples(index=False), total=len(meta), desc="scanning"):
        text = load_text(guid)
        char_len = len(text)
        tokens = tokenize(text)
        emojis = extract_emojis(text)
        hashtags = HASHTAG_RE.findall(text)

        img_path = find_image(guid)
        if img_path is None:
            missing_images.append(guid)
            w = h = None
        else:
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
                    image_sizes.append((w, h))
            except UnidentifiedImageError:
                corrupted_images.append(str(img_path))
                w = h = None

        rows.append({
            "guid": guid,
            "tag": tag,
            "text": text,
            "char_len": char_len,
            "token_len": len(tokens),
            "num_emojis": len(emojis),
            "num_hashtags": len(hashtags),
            "img_path": str(img_path) if img_path else None,
            "img_w": w,
            "img_h": h
        })

    return pd.DataFrame(rows), missing_images, corrupted_images, image_sizes


def plot_hist(series, title, fname, bins=50):
    plt.figure(figsize=(6,4))
    plt.hist(series.dropna(), bins=bins)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / fname)
    plt.close()


def main():
    print("Project dir:", BASE_DIR)
    print("Reading:", TRAIN_META)

    meta = read_train_meta(TRAIN_META)
    print("Loaded samples:", len(meta))
    print(meta.head())

    df, missing_images, corrupted_images, image_sizes = eda(meta)

    stats = {
        "num_samples": len(df),
        "num_missing_images": len(missing_images),
        "num_corrupted_images": len(corrupted_images),
        "avg_char_len": df["char_len"].mean(),
        "median_char_len": df["char_len"].median(),
        "avg_token_len": df["token_len"].mean(),
        "median_token_len": df["token_len"].median(),
    }

    class_counts = df["tag"].value_counts().to_dict()

    print("Stats:", stats)
    print("Class counts:", class_counts)

    df.to_csv(OUTPUT_DIR / "eda_raw.csv", index=False)

    plot_hist(df["char_len"], "Text length (chars)", "char_len_hist.png")
    plot_hist(df["token_len"], "Text length (tokens)", "token_len_hist.png")

    if image_sizes:
        w = [x for x, _ in image_sizes]
        h = [y for _, y in image_sizes]
        plot_hist(pd.Series(w), "Image width", "img_width_hist.png")
        plot_hist(pd.Series(h), "Image height", "img_height_hist.png")

    # class count plot
    plt.figure(figsize=(6,4))
    pd.Series(class_counts).plot(kind="bar")
    plt.title("Class distribution (tag)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_counts.png")
    plt.close()

    with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
        f.write(f"class_counts: {class_counts}\n")

    print("EDA outputs saved to:", OUTPUT_DIR)
    print("Finished.")


if __name__ == "__main__":
    main()
