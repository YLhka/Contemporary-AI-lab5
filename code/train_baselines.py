# train_baselines.py
"""
Train text-only or image-only baselines.

Usage examples:
# text baseline
python train_baselines.py --mode text --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt --data-dir data --model-name bert-base-uncased --epochs 5 --batch-size 16

# image baseline (ResNet50)
python train_baselines.py --mode image --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt --data-dir data --epochs 10 --batch-size 32
"""
import os
import argparse
from pathlib import Path
import json
from typing import List, Dict
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

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

# For text
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW


# For images
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models

# ---------------------------
# label mapping (keep consistent)
# ---------------------------
LABEL2ID = {'positive': 0, 'negative': 1, 'neutral': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------------------------
# Text Dataset
# ---------------------------
class TextDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer_name: str, max_length: int = 128):
        df = pd.read_csv(csv_file, sep='\t')
        # Expect columns: guid, text_clean, tag (if your file uses text_clean name)
        if 'text_clean' in df.columns:
            texts = df['text_clean'].fillna('').astype(str).tolist()
        elif 'text' in df.columns:
            texts = df['text'].fillna('').astype(str).tolist()
        else:
            raise ValueError("No text column found in CSV. Expected 'text_clean' or 'text'.")
        labels = [LABEL2ID[str(t).strip()] for t in df['tag'].tolist()]
        self.texts = texts
        self.labels = labels
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        txt = self.texts[idx]
        enc = self.tokenizer(
            txt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }
        return item

# ---------------------------
# Image Dataset
# ---------------------------
class ImageDataset(Dataset):
    def __init__(self, csv_file: str, data_dir: str, train: bool = True, image_size: int = 224):
        df = pd.read_csv(csv_file, sep='\t')
        # expect columns: guid, img_path, img_exists
        if 'img_path' in df.columns:
            paths = df['img_path'].tolist()
        else:
            # fallback to data/{guid}.jpg
            paths = [str(Path(data_dir) / f"{g}.jpg") for g in df['guid'].astype(str).tolist()]
        labels = [LABEL2ID[str(t).strip()] for t in df['tag'].tolist()]
        self.paths = paths
        self.labels = labels
        self.train = train
        size = image_size
        if train:
            self.transform = T.Compose([
                T.Resize((int(size*1.1), int(size*1.1))),
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((size, size)),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            # fallback: create black image so training doesn't crash
            img = Image.new('RGB', (224,224), (0,0,0))
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'image': img, 'label': label}

# ---------------------------
# Text Model (BERT encoder + MLP)
# ---------------------------
class TextClassifier(nn.Module):
    def __init__(self, model_name: str, hidden_dim: int = 512, dropout: float = 0.1, num_labels: int = 3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.pool = lambda x, mask: x[:,0]  # CLS pooling (AutoModel returns last_hidden_state)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden = out.last_hidden_state  # (B, L, H)
        pooled = self.pool(last_hidden, attention_mask)  # (B, H)
        logits = self.classifier(pooled)
        return logits

# ---------------------------
# Image Model (ResNet50 + MLP)
# ---------------------------
class ImageClassifier(nn.Module):
    def __init__(self, backbone: str = 'resnet50', hidden_dim: int = 512, num_labels: int = 3, pretrained: bool = True):
        super().__init__()
        if backbone == 'resnet50':
            m = models.resnet50(pretrained=pretrained)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
        else:
            raise NotImplementedError("Only resnet50 implemented for now.")
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, x):
        feat = self.backbone(x)  # (B, feat_dim)
        logits = self.classifier(feat)
        return logits

# ---------------------------
# Training / Eval helpers
# ---------------------------
def compute_class_weights(labels: List[int]):
    classes = np.unique(labels)
    # compute_class_weight expects labels as strings or ints of classes
    cw = compute_class_weight(class_weight='balanced', classes=np.array(list(LABEL2ID.values())), y=np.array(labels))
    return torch.tensor(cw, dtype=torch.float)

def train_epoch_text(model, dataloader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0.0
    preds = []
    trues = []
    for batch in tqdm(dataloader, desc='train'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        preds += logits.argmax(dim=1).detach().cpu().tolist()
        trues += labels.detach().cpu().tolist()
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, preds, trues

def eval_epoch_text(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds += logits.argmax(dim=1).detach().cpu().tolist()
            trues += labels.detach().cpu().tolist()
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, preds, trues

def train_epoch_image(model, dataloader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    preds = []
    trues = []
    for batch in tqdm(dataloader, desc='train'):
        imgs = batch['image'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        preds += logits.argmax(dim=1).detach().cpu().tolist()
        trues += labels.detach().cpu().tolist()
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, preds, trues

def eval_epoch_image(model, dataloader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    preds = []
    trues = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval'):
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            logits = model(imgs)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * labels.size(0)
            preds += logits.argmax(dim=1).detach().cpu().tolist()
            trues += labels.detach().cpu().tolist()
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, preds, trues

# ---------------------------
# Main
# ---------------------------
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

        # load train/val csvs
        train_csv = args.train_file
        val_csv = args.val_file

        # compute class weights from train labels
        train_df = pd.read_csv(train_csv, sep='\t')
        train_labels = [LABEL2ID[str(t).strip()] for t in train_df['tag'].tolist()]
        class_weights = compute_class_weights(train_labels).to(device)
        print(f"Class weights: {class_weights.tolist()}")

        if args.mode == 'text':
            tokenizer_name = args.model_name
            train_ds = TextDataset(train_csv, tokenizer_name=tokenizer_name, max_length=args.max_length)
            val_ds = TextDataset(val_csv, tokenizer_name=tokenizer_name, max_length=args.max_length)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

            model = TextClassifier(model_name=tokenizer_name, hidden_dim=args.hidden_dim).to(device)

            # optimizer: smaller lr for encoder, larger lr for classifier
            no_decay = ['bias', 'LayerNorm.weight']
            enc_params = list(model.encoder.named_parameters())
            enc_optimizer_grouped_parameters = [
                {'params': [p for n,p in model.encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr': args.lr_encoder},
                {'params': [p for n,p in model.encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.lr_encoder},
            ]
            # classifier params
            cls_params = [p for n,p in model.classifier.named_parameters()]
            optimizer = AdamW(enc_optimizer_grouped_parameters + [{'params': cls_params, 'lr': args.lr}], lr=1e-5)
            total_steps = len(train_loader) * args.epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)

            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

            # 早停机制
            best_f1 = -1.0
            patience = args.patience
            counter = 0

            # 记录每个epoch的指标
            metrics = []

            for epoch in range(1, args.epochs + 1):
                print(f"===== Epoch {epoch} =====")
                train_loss, train_preds, train_trues = train_epoch_text(model, train_loader, optimizer, scheduler, device, loss_fn)
                val_loss, val_preds, val_trues = eval_epoch_text(model, val_loader, device, loss_fn)
                val_macro_f1 = f1_score(val_trues, val_preds, average='macro')
                val_weighted_f1 = f1_score(val_trues, val_preds, average='weighted')
                val_acc = accuracy_score(val_trues, val_preds)
                
                # 打印详细指标
                print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                print(f"Val macro-F1: {val_macro_f1:.4f} | Val weighted-F1: {val_weighted_f1:.4f} | Val acc: {val_acc:.4f}")
                report = classification_report(val_trues, val_preds, target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))], digits=4)
                cm = confusion_matrix(val_trues, val_preds)
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
                    'val_macro_f1': val_macro_f1,
                    'val_weighted_f1': val_weighted_f1,
                    'val_acc': val_acc
                })
                
                # 保存当前epoch的报告
                epoch_report_file = os.path.join(args.output_dir, f'epoch_{epoch}_report.txt')
                with open(epoch_report_file, 'w', encoding='utf-8') as f:
                    f.write(f"Epoch {epoch} Report\n")
                    f.write(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}\n")
                    f.write(f"Val macro-F1: {val_macro_f1:.4f} | Val weighted-F1: {val_weighted_f1:.4f} | Val acc: {val_acc:.4f}\n\n")
                    f.write("Classification Report:\n")
                    f.write(report + '\n\n')
                    f.write("Confusion Matrix:\n")
                    f.write(str(cm) + '\n')
                
                # 早停和模型保存
                if val_macro_f1 > best_f1:
                    best_f1 = val_macro_f1
                    torch.save(model.state_dict(), args.output_path)
                    print(f"Saved best model: {args.output_path}")
                    counter = 0
                else:
                    counter += 1
                    print(f"Early stopping counter: {counter}/{patience}")
                    if counter >= patience:
                        print("Early stopping triggered!")
                        break
            print(f"Best val macro-F1: {best_f1:.4f}")

        elif args.mode == 'image':
            train_ds = ImageDataset(train_csv, data_dir=args.data_dir, train=True, image_size=args.image_size)
            val_ds = ImageDataset(val_csv, data_dir=args.data_dir, train=False, image_size=args.image_size)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

            model = ImageClassifier(backbone='resnet50', hidden_dim=args.hidden_dim, pretrained=True).to(device)
            # optionally freeze backbone initial layers to speed up
            if args.freeze_backbone:
                for name, p in model.backbone.named_parameters():
                    if not any([k in name for k in ['layer4', 'fc']]):
                        p.requires_grad = False
                print("Backbone partially frozen (only layer4 and fc trainable).")

            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)

            # 早停机制
            best_f1 = -1.0
            patience = args.patience
            counter = 0

            # 记录每个epoch的指标
            metrics = []

            for epoch in range(1, args.epochs+1):
                print(f"===== Epoch {epoch} =====")
                train_loss, train_preds, train_trues = train_epoch_image(model, train_loader, optimizer, device, loss_fn)
                val_loss, val_preds, val_trues = eval_epoch_image(model, val_loader, device, loss_fn)
                val_macro_f1 = f1_score(val_trues, val_preds, average='macro')
                val_weighted_f1 = f1_score(val_trues, val_preds, average='weighted')
                val_acc = accuracy_score(val_trues, val_preds)
                
                # 打印详细指标
                print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
                print(f"Val macro-F1: {val_macro_f1:.4f} | Val weighted-F1: {val_weighted_f1:.4f} | Val acc: {val_acc:.4f}")
                report = classification_report(val_trues, val_preds, target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))], digits=4)
                cm = confusion_matrix(val_trues, val_preds)
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
                    'val_macro_f1': val_macro_f1,
                    'val_weighted_f1': val_weighted_f1,
                    'val_acc': val_acc
                })
                
                # 保存当前epoch的报告
                epoch_report_file = os.path.join(args.output_dir, f'epoch_{epoch}_report.txt')
                with open(epoch_report_file, 'w', encoding='utf-8') as f:
                    f.write(f"Epoch {epoch} Report\n")
                    f.write(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}\n")
                    f.write(f"Val macro-F1: {val_macro_f1:.4f} | Val weighted-F1: {val_weighted_f1:.4f} | Val acc: {val_acc:.4f}\n\n")
                    f.write("Classification Report:\n")
                    f.write(report + '\n\n')
                    f.write("Confusion Matrix:\n")
                    f.write(str(cm) + '\n')
                
                # 早停和模型保存
                if val_macro_f1 > best_f1:
                    best_f1 = val_macro_f1
                    torch.save(model.state_dict(), args.output_path)
                    print(f"Saved best model: {args.output_path}")
                    counter = 0
                else:
                    counter += 1
                    print(f"Early stopping counter: {counter}/{patience}")
                    if counter >= patience:
                        print("Early stopping triggered!")
                        break
            print(f"Best val macro-F1: {best_f1:.4f}")

        else:
            raise ValueError("mode must be 'text' or 'image'")

        # 保存指标到CSV文件
        metrics_df = pd.DataFrame(metrics)
        metrics_file = os.path.join(args.output_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Saved metrics to: {metrics_file}")

        # load best model then evaluate on val and print final report
        print("\n===== Final Evaluation =====")
        if args.mode == 'text':
            model.load_state_dict(torch.load(args.output_path, map_location=device))
            _, val_preds, val_trues = eval_epoch_text(model, val_loader, device, loss_fn)
        else:
            model.load_state_dict(torch.load(args.output_path, map_location=device))
            _, val_preds, val_trues = eval_epoch_image(model, val_loader, device, loss_fn)

        val_macro_f1 = f1_score(val_trues, val_preds, average='macro')
        val_weighted_f1 = f1_score(val_trues, val_preds, average='weighted')
        val_acc = accuracy_score(val_trues, val_preds)
        report = classification_report(val_trues, val_preds, target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))], digits=4)
        cm = confusion_matrix(val_trues, val_preds)
        
        print(f"Final val macro-F1: {val_macro_f1:.4f} | Final val weighted-F1: {val_weighted_f1:.4f} | Final val acc: {val_acc:.4f}")
        print("\nFinal Classification Report:")
        print(report)
        print("\nFinal Confusion Matrix:")
        print(cm)
        
        # 保存最终报告
        final_report_file = os.path.join(args.output_dir, 'final_report.txt')
        with open(final_report_file, 'w', encoding='utf-8') as f:
            f.write("Final Evaluation Report\n")
            f.write(f"Best val macro-F1: {best_f1:.4f}\n")
            f.write(f"Final val macro-F1: {val_macro_f1:.4f} | Final val weighted-F1: {val_weighted_f1:.4f} | Final val acc: {val_acc:.4f}\n\n")
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
    p.add_argument('--mode', type=str, choices=['text','image'], required=True)
    p.add_argument('--train-file', type=str, default='cleaned/train_clean.txt')
    p.add_argument('--val-file', type=str, default='cleaned/val_clean.txt')
    p.add_argument('--data-dir', type=str, default='data')
    # common
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--hidden-dim', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)  # classifier lr or image baseline lr
    p.add_argument('--output-path', type=str, default='best_baseline.pt')
    p.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    # text-specific
    p.add_argument('--model-name', type=str, default='bert-base-uncased')
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--lr-encoder', type=float, default=2e-5)
    # image-specific
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--freeze-backbone', action='store_true')
    # Output directory
    p.add_argument('--output-dir', type=str, default='results/baselines', help='Directory to save results')
    args = p.parse_args()
    
    # 根据模式创建不同的输出目录
    args.output_dir = os.path.join(args.output_dir, args.mode)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 修改output_path为包含模式信息的文件名，直接放在外面
    args.output_path = f'best_{args.mode}_baseline.pt'
    main(args)
