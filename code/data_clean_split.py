#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_clean_split.py

功能（按要求重写）:
- 读取 train.txt（支持 \t 或 , 分隔，期望至少包含 guid 和 tag 列）
- 为每个 guid 检查 data/{guid}.jpg/.jpeg/.png 是否存在
- 检测图片是否损坏（PIL verify），损坏样本将被记录并可选择移除
- 将缺失图片标记为 image-less（保留样本，但记录在 image_less.csv，便于消融）
- 文本清洗:
    - 去 HTML 标签
    - 去 URL
    - collapse 多余空白
    - 去掉控制字符 / 无意义字符（但保留 emoji、#、@）
    - 可选统一小写（--lowercase）
- 若 train.txt 中没有 text 列，可以:
    - 使用 --use-guid-as-text 将 guid 填入 text（占位）
    - 或使用 --try-load-text-files 从 data/{guid}.txt 尝试加载文本
- 支持 Stratified 80/20 split 或 Stratified K-Fold（--kfold）
- 输出:
    cleaned/train_clean.txt, cleaned/val_clean.txt  或 folds
    cleaned/removed_samples.csv  (损坏且被移除)
    cleaned/image_less.csv      (缺图但保留)
    cleaned/split_indices.json  (保存 guid 列表，方便复现实验)
- 可选: 使用 HuggingFace tokenizer 对清洗文本做 tokenization（仅当安装 transformers 时启用 --tokenize）
"""

import re
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from PIL import Image, UnidentifiedImageError

# -------------------------
# 文本清洗工具函数
# -------------------------
HTML_RE = re.compile(r'<.*?>', flags=re.S)
URL_RE = re.compile(r'https?://\S+|www\.\S+')
# 控制字符（不可见）删除
CONTROL_CHARS_RE = re.compile(r'[\u0000-\u001f\u007f-\u009f]')
# 连续标点压缩（超过3个变为1个）
MULTI_PUNCT_RE = re.compile(r'([^\w\s#@])\1{2,}')
# 保留 # 和 @，保留 emoji（emoji 属于 unicode 的符号范围，通常不会被下面的删除逻辑删掉）
# 若需更精确的 emoji 识别，可用 emoji 模块，但这里尽量不依赖额外包

def clean_text(text: str, lowercase: bool = False) -> str:
    if text is None:
        return ''
    s = str(text)
    # 去 HTML
    s = HTML_RE.sub(' ', s)
    # 去 URL
    s = URL_RE.sub('', s)
    # 删除控制字符
    s = CONTROL_CHARS_RE.sub('', s)
    # 将长连标点缩短
    s = MULTI_PUNCT_RE.sub(r'\1', s)
    # collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    if lowercase:
        s = s.lower()
    return s

# -------------------------
# 图片检测工具
# -------------------------
IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']

def find_image_path(data_dir: Path, guid: str) -> Optional[Path]:
    for ext in IMAGE_EXTS:
        p = data_dir / f"{guid}{ext}"
        if p.exists():
            return p
    return None

def is_image_ok(img_path: Path) -> bool:
    try:
        with Image.open(img_path) as im:
            im.verify()  # detect truncated/corrupted images in many cases
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False

# -------------------------
# 自动识别分隔符并读文件（尝试 tab, comma）
# -------------------------
def read_table_auto(path: Path) -> pd.DataFrame:
    for sep in ['\t', ',']:
        try:
            df = pd.read_csv(path, sep=sep)
            cols = [c.lower() for c in df.columns]
            if 'guid' in cols and 'tag' in cols:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception:
            continue
    # 尝试无 header 的读取（至少两列）
    try:
        df = pd.read_csv(path, header=None, engine='python')
        if df.shape[1] >= 2:
            names = ['guid', 'tag'] + [f'col{i}' for i in range(3, df.shape[1]+1)]
            df = pd.read_csv(path, header=None, names=names, engine='python')
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception:
        pass
    raise ValueError(f"无法解析 {path}，请检查文件格式，需包含 guid 和 tag 列。")

# -------------------------
# 主流程
# -------------------------
def main(args):
    train_file = Path(args.train_file)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"读取: {train_file}")
    df = read_table_auto(train_file)
    print(f"原始样本数: {len(df)}")
    # 统一小写列名
    df.columns = [c.lower() for c in df.columns]

    # 确保 guid, tag
    if 'guid' not in df.columns or 'tag' not in df.columns:
        raise RuntimeError("输入文件必须包含 'guid' 和 'tag' 列。")

    # text 处理策略
    if 'text' not in df.columns:
        if args.use_guid_as_text:
            df['text'] = df['guid'].astype(str)
            print("没有 text 列 -> 使用 guid 作为占位文本 (--use-guid-as-text)。")
        else:
            df['text'] = ''
            print("没有 text 列 -> text 列为空（若有 data/{guid}.txt 可用 --try-load-text-files 加载）。")

    # 尝试从 data/{guid}.txt 加载文本（当 text 为空时）
    if args.try_load_text_files:
        def load_text_from_file(row):
            if row['text'] and str(row['text']).strip():
                return row['text']
            tpath = data_dir / f"{row['guid']}.txt"
            if tpath.exists():
                try:
                    return tpath.read_text(encoding='utf-8').strip()
                except Exception:
                    try:
                        return tpath.read_text(encoding='gbk').strip()
                    except Exception:
                        return ''
            return row['text']
        df['text'] = df.apply(load_text_from_file, axis=1)
        print("已尝试从 data/{guid}.txt 加载文本（若存在）。")

    # 文本清洗
    df['text_clean'] = df['text'].apply(lambda t: clean_text(t, lowercase=args.lowercase))

    # 对图片做存在/损坏检测
    img_exists = []
    img_ok = []
    img_path_list = []
    for guid in df['guid']:
        p = find_image_path(data_dir, str(guid))
        if p is None:
            img_exists.append(False)
            img_ok.append(False)
            img_path_list.append(None)
        else:
            img_exists.append(True)
            ok = is_image_ok(p)
            img_ok.append(bool(ok))
            img_path_list.append(str(p.resolve()))
    df['img_exists'] = img_exists
    df['img_ok'] = img_ok
    df['img_path'] = img_path_list

    # 记录缺图样本（image-less - 保留）
    image_less_df = df[~df['img_exists']].copy()
    if len(image_less_df) > 0:
        image_less_df[['guid', 'tag']].to_csv(out_dir / 'image_less.csv', index=False)
        print(f"缺图 (image-less) 数量: {len(image_less_df)} -> 已保存 {out_dir/'image_less.csv'}")
    else:
        print("没有缺图样本。")

    # 记录损坏图片样本（默认移除）
    corrupted_df = df[df['img_exists'] & (~df['img_ok'])].copy()
    if len(corrupted_df) > 0:
        corrupted_df[['guid', 'tag', 'img_path']].to_csv(out_dir / 'removed_corrupted_images.csv', index=False)
        print(f"检测到损坏图片: {len(corrupted_df)} -> 已保存 {out_dir/'removed_corrupted_images.csv'}")
        if args.remove_corrupted:
            print("将损坏图片样本从数据集中移除（remove_corrupted=True）。")
            df = df[~(df['img_exists'] & (~df['img_ok']))].copy()
        else:
            # 若不移除，标记为 image-less（视为无图）
            df.loc[df['img_exists'] & (~df['img_ok']), 'img_exists'] = False
            df.loc[df['img_exists'] & (~df['img_ok']), 'img_ok'] = False
            print("损坏图片样本将被标记为 image-less（保留样本，用于消融）。")
    else:
        print("没有损坏图片。")

    # 最终保留样本（包含 image-less），但不包含明确移除的样本
    kept = df.copy()
    print(f"最终保留样本数: {len(kept)}")
    print("各类分布（保留样本）:")
    print(kept['tag'].value_counts().to_dict())

    # 检查每类是否至少有 k 折或能做 stratify 划分
    min_count = kept['tag'].value_counts().min()
    if args.kfold and args.kfold > 1:
        if min_count < args.kfold:
            raise RuntimeError(f"某些类别样本数小于 k={args.kfold}，无法做 StratifiedKFold。每类至少需要 k 个样本。")
    else:
        # stratified train_test_split 要求每类至少 2 个样本，通常没有问题
        pass

    # 保存清洗后的全部表（包含 text_clean, img_path, img flags）
    kept.to_csv(out_dir / 'all_cleaned_metadata.csv', index=False)
    print(f"已保存全部清洗后元数据到 {out_dir/'all_cleaned_metadata.csv'}")

    split_indices = {}

    # K-fold 或 80/20
    if args.kfold and args.kfold > 1:
        k = args.kfold
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=args.seed)
        X = kept.index.values
        y = kept['tag'].values
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            train_df = kept.iloc[train_idx].reset_index(drop=True)
            val_df = kept.iloc[val_idx].reset_index(drop=True)
            train_path = out_dir / f'fold{fold_idx}_train_clean.txt'
            val_path = out_dir / f'fold{fold_idx}_val_clean.txt'
            train_df[['guid', 'text_clean', 'tag', 'img_path', 'img_exists', 'img_ok']].to_csv(
                train_path, sep='\t', index=False)
            val_df[['guid', 'text_clean', 'tag', 'img_path', 'img_exists', 'img_ok']].to_csv(
                val_path, sep='\t', index=False)
            print(f"Fold {fold_idx}: saved train={len(train_df)} -> {train_path}, val={len(val_df)} -> {val_path}")
            split_indices[f'fold{fold_idx}'] = {
                'train': train_df['guid'].tolist(),
                'val': val_df['guid'].tolist()
            }
    else:
        # stratified 80/20 默认
        train_df, val_df = train_test_split(
            kept,
            test_size=args.test_size,
            stratify=kept['tag'],
            random_state=args.seed
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        train_path = out_dir / 'train_clean.txt'
        val_path = out_dir / 'val_clean.txt'
        train_df[['guid', 'text_clean', 'tag', 'img_path', 'img_exists', 'img_ok']].to_csv(
            train_path, sep='\t', index=False)
        val_df[['guid', 'text_clean', 'tag', 'img_path', 'img_exists', 'img_ok']].to_csv(
            val_path, sep='\t', index=False)
        print(f"Saved: {train_path} ({len(train_df)}) and {val_path} ({len(val_df)})")
        split_indices['train'] = train_df['guid'].tolist()
        split_indices['val'] = val_df['guid'].tolist()

    # 保存 split indices
    with open(out_dir / 'split_indices.json', 'w', encoding='utf-8') as f:
        json.dump(split_indices, f, ensure_ascii=False, indent=2)

    # 若用户要求 tokenization（需 transformers 安装）
    if args.tokenize:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            print(f"使用 tokenizer: {args.tokenizer_name} 对清洗文本进行 tokenization（并保存 token ids 到 CSV，注意可能很大）。")
            # 仅示例：对 train_clean 做 tokenization 并保存 tokens 长度统计
            tokenized = []
            for t in pd.read_csv(out_dir / 'train_clean.txt', sep='\t')['text_clean']:
                enc = tokenizer.encode_plus(str(t), truncation=True, max_length=args.max_length, padding=False)
                tokenized.append({'orig_text': t, 'input_ids_len': len(enc['input_ids'])})
            tok_df = pd.DataFrame(tokenized)
            tok_df.to_csv(out_dir / 'train_tokenization_stats.csv', index=False)
            print(f"tokenization 完成并保存到 {out_dir/'train_tokenization_stats.csv'}")
        except Exception as e:
            print("tokenize 失败（可能未安装 transformers 或 tokenizer 名称错误）:", e)

    print("完成数据清洗与划分。输出目录:", out_dir)

# -------------------------
# CLI 参数
# -------------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser(description="数据清洗与分层划分脚本（保留 image-less，用于消融）")
    p.add_argument('--train-file', type=str, default='train.txt', help='训练文件路径（包含 guid 和 tag）')
    p.add_argument('--data-dir', type=str, default='data', help='图片与可能的单独文本文件所在目录（data/{guid}.jpg, data/{guid}.txt）')
    p.add_argument('--output-dir', type=str, default='cleaned', help='输出目录')
    p.add_argument('--test-size', type=float, default=0.2, help='验证集比例（若不使用 kfold）')
    p.add_argument('--kfold', type=int, default=0, help='若 >1 则进行 StratifiedKFold（例如 5）')
    p.add_argument('--seed', type=int, default=42, help='随机种子')
    p.add_argument('--lowercase', action='store_true', help='是否将文本统一小写（谨慎）')
    p.add_argument('--try-load-text-files', action='store_true', help='若 text 为空，尝试从 data/{guid}.txt 加载文本')
    p.add_argument('--use-guid-as-text', action='store_true', help='若没有 text 列，使用 guid 作为占位文本')
    p.add_argument('--remove-corrupted', action='store_true', help='是否从数据中移除损坏图片样本（否则将其标记为 image-less）')
    p.add_argument('--tokenize', action='store_true', help='是否尝试用 transformers tokenizer 对 train_clean 文本做 tokenization（需安装 transformers）')
    p.add_argument('--tokenizer-name', type=str, default='bert-base-uncased', help='tokenizer 名称（HuggingFace），当 --tokenize 时使用')
    p.add_argument('--max-length', type=int, default=128, help='tokenizer 最大长度（当 --tokenize 时使用）')
    args = p.parse_args()
    main(args)
