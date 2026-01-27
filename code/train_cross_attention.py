# train_cross_attention.py
"""
Cross-Attention / Joint Multimodal Transformer for Text+Image fusion.

Usage examples:
# joint fusion (concatenate tokens -> transformer)
python train_cross_attention.py --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt \
    --data-dir data --fusion joint --text-model bert-base-uncased --image-model google/vit-base-patch16-224 \
    --epochs 6 --batch-size 8 --lr 2e-5 --output best_cross.pt

# cross-attention fusion (text <-> image cross attention)
python train_cross_attention.py --fusion cross --epochs 6 --batch-size 8 --freeze-text --freeze-image
"""
import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, ViTModel, ViTImageProcessor, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import random

# 固定随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

LABEL2ID = {'positive': 0, 'negative': 1, 'neutral': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# -------------------------
# Dataset: uses tokenizer + ViTImageProcessor
# -------------------------
class CrossModalDataset(Dataset):
    def __init__(self, csv_file: str, data_dir: str,
                 tokenizer_name: str, image_processor_name: str,
                 max_length: int = 128, image_size: int = 224, train: bool = True):
        df = pd.read_csv(csv_file, sep='\t')
        self.guids = df['guid'].astype(str).tolist()
        # prefer text_clean then text
        if 'text_clean' in df.columns:
            self.texts = df['text_clean'].fillna('').astype(str).tolist()
        elif 'text' in df.columns:
            self.texts = df['text'].fillna('').astype(str).tolist()
        else:
            self.texts = [str(g) for g in self.guids]

        if 'img_path' in df.columns:
            self.paths = df['img_path'].fillna('').tolist()
        else:
            self.paths = [str(Path(data_dir) / f"{g}.jpg") for g in self.guids]

        self.labels = [LABEL2ID[str(t).strip()] for t in df['tag'].tolist()]

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.image_processor = ViTImageProcessor.from_pretrained(image_processor_name)
        self.max_length = max_length
        self.image_size = image_size
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        path = self.paths[idx]
        if path and Path(path).exists():
            try:
                image = Image.open(path).convert('RGB')
            except Exception:
                image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        else:
            image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))

        text_enc = self.tokenizer(text,
                                  truncation=True,
                                  padding='max_length',
                                  max_length=self.max_length,
                                  return_tensors='pt')
        img_enc = self.image_processor(images=image, return_tensors='pt')

        item = {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'pixel_values': img_enc['pixel_values'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

# -------------------------
# Small Transformer blocks (FFN, LayerNorm helpers)
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.net(x))

# -------------------------
# Cross-Attention Layer: text attends to image and image attends to text (optional self-attn)
# -------------------------
class CrossAttentionLayer(nn.Module):
    def __init__(self, dim, nhead=8, dropout=0.1, ff_dim=2048, self_attend=False):
        super().__init__()
        self.self_attend = self_attend
        # Multihead for cross: queries from A, keys/vals from B
        self.cross_attn_t2i = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ff_t = FeedForward(dim, ff_dim, dropout)

        self.cross_attn_i2t = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
        self.ff_i = FeedForward(dim, ff_dim, dropout)

        if self_attend:
            self.self_attn_t = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)
            self.self_attn_i = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True)

    def forward(self, text_tokens, image_tokens, text_mask=None, image_mask=None):
        # text_tokens: (B, Lt, D), image_tokens: (B, Li, D)
        # optionally add self-attention first
        if self.self_attend:
            t_res, _ = self.self_attn_t(text_tokens, text_tokens, text_tokens, key_padding_mask=text_mask)
            i_res, _ = self.self_attn_i(image_tokens, image_tokens, image_tokens, key_padding_mask=image_mask)
            text_tokens = text_tokens + t_res
            image_tokens = image_tokens + i_res

        # text attends to image
        t2i_res, _ = self.cross_attn_t2i(text_tokens, image_tokens, image_tokens,
                                        key_padding_mask=image_mask, need_weights=False)
        text_tokens = text_tokens + t2i_res
        text_tokens = self.ff_t(text_tokens)

        # image attends to text
        i2t_res, _ = self.cross_attn_i2t(image_tokens, text_tokens, text_tokens,
                                         key_padding_mask=text_mask, need_weights=False)
        image_tokens = image_tokens + i2t_res
        image_tokens = self.ff_i(image_tokens)

        return text_tokens, image_tokens

# -------------------------
# Joint Transformer (concatenate tokens -> encoder)
# -------------------------
class JointTransformer(nn.Module):
    def __init__(self, dim, n_layers=4, nhead=8, ff_dim=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, seq, attention_mask=None):
        # seq: (B, L, D)
        out = self.encoder(seq, src_key_padding_mask=attention_mask)  # mask shape (B, L)
        return self.norm(out)

# -------------------------
# Multimodal model
# -------------------------
class CrossModalModel(nn.Module):
    def __init__(self,
                 text_model_name='bert-base-uncased',
                 image_model_name='google/vit-base-patch16-224',
                 hidden_dim=768,
                 proj_dim=512,
                 fusion='cross',   # 'cross' or 'joint'
                 n_layers=2,
                 nhead=8,
                 ff_dim=2048,
                 freeze_text=False,
                 freeze_image=False,
                 num_labels=3):
        super().__init__()
        self.fusion = fusion
        # encoders
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.image_encoder = ViTModel.from_pretrained(image_model_name)

        text_dim = self.text_encoder.config.hidden_size
        img_dim = self.image_encoder.config.hidden_size

        # project both to common hidden_dim
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.img_proj = nn.Linear(img_dim, hidden_dim)

        if fusion == 'joint':
            # CLS token for joint sequence
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.pos_emb = nn.Parameter(torch.randn(1, 512, hidden_dim))  # max seq len support
            self.joint_transformer = JointTransformer(dim=hidden_dim, n_layers=n_layers, nhead=nhead, ff_dim=ff_dim)
        else:
            # cross-attention layers
            self.cross_layers = nn.ModuleList([CrossAttentionLayer(dim=hidden_dim, nhead=nhead, dropout=0.1, ff_dim=ff_dim, self_attend=False) for _ in range(n_layers)])

        # classifier: pool => MLP
        self.pooler = nn.AdaptiveAvgPool1d(1)  # fallback pooling if needed
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 1),  # use text pooled OR combined
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(proj_dim, num_labels)
        )

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, pixel_values):
        # input_ids: (B, Lt), pixel_values: (B, C, H, W)
        # 1) get raw embeddings
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        t_tokens = t_out.last_hidden_state  # (B, Lt, text_dim)

        v_out = self.image_encoder(pixel_values=pixel_values, return_dict=True)
        v_tokens = v_out.last_hidden_state  # (B, Lp, img_dim)  (ViT includes class token at index 0)

        # 2) project to common dim
        t = self.text_proj(t_tokens)  # (B, Lt, D)
        v = self.img_proj(v_tokens)   # (B, Lp, D)

        if self.fusion == 'joint':
            # build sequence: [CLS] + text_tokens + image_tokens
            B = t.shape[0]
            cls = self.cls_token.expand(B, -1, -1)  # (B,1,D)
            seq = torch.cat([cls, t, v], dim=1)  # (B, 1+Lt+Lp, D)
            # optional positional embedding (truncate/pad pos_emb)
            L = seq.shape[1]
            if L <= self.pos_emb.size(1):
                seq = seq + self.pos_emb[:, :L, :].to(seq.device)
            out = self.joint_transformer(seq)  # (B, L, D)
            cls_out = out[:, 0, :]  # (B, D)
            pooled = cls_out
        else:
            # cross-attention stacks
            text_tokens = t
            image_tokens = v
            # no padding mask handling here for simplicity (can be extended)
            for layer in self.cross_layers:
                text_tokens, image_tokens = layer(text_tokens, image_tokens)
            # pool text tokens (mean)
            pooled = text_tokens.mean(dim=1)  # (B, D)
            # you could also combine image pooled: pooled = 0.5*(text.mean+image.mean) or concat
            # we use text pooled for classifier (could be changed)
        logits = self.classifier(pooled)
        return logits

# -------------------------
# Train / Eval loops
# -------------------------
def compute_class_weights(train_csv):
    df = pd.read_csv(train_csv, sep='\t')
    labels = [LABEL2ID[str(x).strip()] for x in df['tag'].tolist()]
    classes = np.array(list(LABEL2ID.values()))
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=np.array(labels))
    return torch.tensor(cw, dtype=torch.float)

def train_epoch(model, loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0.
    for batch in tqdm(loader, desc='train'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attn, pixel_values=pixel_values)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)

def eval_model(model, loader, device, loss_fn=None):
    model.eval()
    preds, trues = [], []
    total_loss = 0.
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval'):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn, pixel_values=pixel_values)
            if loss_fn is not None:
                total_loss += loss_fn(logits, labels).item() * labels.size(0)
            preds += logits.argmax(dim=1).cpu().tolist()
            trues += labels.cpu().tolist()
    
    # 计算指标
    macro = f1_score(trues, preds, average='macro')
    weighted = f1_score(trues, preds, average='weighted')
    acc = accuracy_score(trues, preds)
    report = classification_report(trues, preds, target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))], digits=4)
    cm = confusion_matrix(trues, preds)
    avg_loss = (total_loss / len(loader.dataset)) if loss_fn is not None else None
    
    return avg_loss, macro, weighted, acc, report, cm, trues, preds

# -------------------------
# CLI main
# -------------------------
def main(args):
    # 创建日志文件
    log_file = os.path.join(args.output_dir, 'training_log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {' '.join(os.sys.argv)}\n\n")

    # 重定向输出到日志文件和终端
    class Tee:
        def __init__(self, filename):
            self.file = open(filename, 'a', encoding='utf-8')
            self.stdout = os.sys.stdout
            os.sys.stdout = self
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
        def flush(self):
            self.file.flush()
            self.stdout.flush()
        def close(self):
            os.sys.stdout = self.stdout
            self.file.close()

    tee = Tee(log_file)

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        print(f"PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # dataset
        ds_train = CrossModalDataset(args.train_file, args.data_dir, tokenizer_name=args.text_model, image_processor_name=args.image_model,
                                     max_length=args.max_length, image_size=args.image_size, train=True)
        ds_val = CrossModalDataset(args.val_file, args.data_dir, tokenizer_name=args.text_model, image_processor_name=args.image_model,
                                   max_length=args.max_length, image_size=args.image_size, train=False)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        # class weights
        cw = compute_class_weights(args.train_file).to(device)
        print(f"Class weights: {cw.tolist()}")

        model = CrossModalModel(text_model_name=args.text_model, image_model_name=args.image_model,
                                hidden_dim=args.hidden_dim, proj_dim=args.proj_dim,
                                fusion=args.fusion, n_layers=args.n_layers, nhead=args.nhead,
                                ff_dim=args.ff_dim, freeze_text=args.freeze_text, freeze_image=args.freeze_image,
                                num_labels=len(LABEL2ID)).to(device)

        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

        loss_fn = nn.CrossEntropyLoss(weight=cw)

        # 早停机制
        best_f1 = -1.0
        patience = args.patience
        counter = 0

        # 记录每个epoch的指标
        metrics = []

        for epoch in range(1, args.epochs + 1):
            print(f"===== Epoch {epoch} =====")
            train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
            val_loss, val_f1, val_f1_weighted, val_acc, report, cm, trues, preds = eval_model(model, val_loader, device, loss_fn)
            
            # 打印详细指标
            print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
            print(f"Val macro-F1: {val_f1:.4f} | Val weighted-F1: {val_f1_weighted:.4f} | Val acc: {val_acc:.4f}")
            print("\nClassification Report:")
            print(report)
            print("\nConfusion Matrix:")
            print(cm)
            print()
            
            # 记录指标
            metrics.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_macro_f1': val_f1,
                'val_weighted_f1': val_f1_weighted,
                'val_acc': val_acc
            })
            
            # 保存当前epoch的报告
            epoch_report_file = os.path.join(args.output_dir, f'epoch_{epoch}_report.txt')
            with open(epoch_report_file, 'w', encoding='utf-8') as f:
                f.write(f"Epoch {epoch} Report\n")
                f.write(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}\n")
                f.write(f"Val macro-F1: {val_f1:.4f} | Val weighted-F1: {val_f1_weighted:.4f} | Val acc: {val_acc:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(report + '\n\n')
                f.write("Confusion Matrix:\n")
                f.write(str(cm) + '\n')
            
            # 早停和模型保存
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), args.output)
                print(f"Saved best model: {args.output}")
                counter = 0
            else:
                counter += 1
                print(f"Early stopping counter: {counter}/{patience}")
                if counter >= patience:
                    print("Early stopping triggered!")
                    break

        # 保存指标到CSV文件
        metrics_df = pd.DataFrame(metrics)
        metrics_file = os.path.join(args.output_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics to: {metrics_file}")

        # final report
        print("\n===== Final Evaluation =====")
        print(f"Best val macro-F1: {best_f1:.4f}")
        model.load_state_dict(torch.load(args.output, map_location=device))
        _, val_f1, val_f1_weighted, val_acc, report, cm, trues, preds = eval_model(model, val_loader, device, loss_fn)
        print(f"Final val macro-F1: {val_f1:.4f} | Final val weighted-F1: {val_f1_weighted:.4f} | Final val acc: {val_acc:.4f}")
        print("\nFinal Classification Report:")
        print(report)
        print("\nFinal Confusion Matrix:")
        print(cm)
        
        # 保存最终报告
        final_report_file = os.path.join(args.output_dir, 'final_report.txt')
        with open(final_report_file, 'w', encoding='utf-8') as f:
            f.write("Final Evaluation Report\n")
            f.write(f"Best val macro-F1: {best_f1:.4f}\n")
            f.write(f"Final val macro-F1: {val_f1:.4f} | Final val weighted-F1: {val_f1_weighted:.4f} | Final val acc: {val_acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report + '\n\n')
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + '\n')
        print(f"Saved final report to: {final_report_file}")
        
        print(f"\nAll results saved to: {args.output_dir}")
    finally:
        tee.close()

if __name__ == '__main__':
    import time
    p = argparse.ArgumentParser()
    p.add_argument('--train-file', type=str, default='cleaned/train_clean.txt')
    p.add_argument('--val-file', type=str, default='cleaned/val_clean.txt')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--text-model', type=str, default='bert-base-uncased')
    p.add_argument('--image-model', type=str, default='google/vit-base-patch16-224')
    p.add_argument('--fusion', type=str, choices=['joint', 'cross'], default='cross', help='fusion type: joint concat or cross-attention')
    p.add_argument('--hidden-dim', type=int, default=512, help='common hidden dim after projection')
    p.add_argument('--proj-dim', type=int, default=512, help='classifier hidden dim')
    p.add_argument('--n_layers', type=int, default=2, help='number of transformer/cross layers')
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--ff_dim', type=int, default=2048)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=2e-5)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--freeze-text', action='store_true')
    p.add_argument('--freeze-image', action='store_true')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output', type=str, default='best_cross_modal.pt')
    p.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    p.add_argument('--output-dir', type=str, default='results/cross_attention', help='Directory to save results')
    args = p.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
