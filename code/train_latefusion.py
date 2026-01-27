import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import pandas as pd
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

# -------------------------
# Dataset
# -------------------------
LABEL2ID = {'positive': 0, 'negative': 1, 'neutral': 2}

class FusionDataset(Dataset):
    def __init__(self, csv_file, data_dir, tokenizer_name='bert-base-uncased', max_length=128, image_size=224, train=True):
        df = pd.read_csv(csv_file, sep='\t')
        self.guids = df['guid'].astype(str).tolist()
        self.texts = df['text_clean'].fillna('').astype(str).tolist()
        self.paths = [str(os.path.join(data_dir, f"{g}.jpg")) for g in self.guids]
        self.labels = [LABEL2ID[t.strip()] for t in df['tag'].tolist()]

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.train = train
        size = image_size

        if train:
            self.img_transform = T.Compose([
                T.Resize((int(size*1.05), int(size*1.05))),
                T.RandomResizedCrop(size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])
        else:
            self.img_transform = T.Compose([
                T.Resize((size, size)),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = enc['input_ids'].squeeze(0)
        attn = enc['attention_mask'].squeeze(0)

        img_path = self.paths[idx]
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                img = Image.new('RGB', (224,224), (0,0,0))
        else:
            img = Image.new('RGB', (224,224), (0,0,0))
        img_tensor = self.img_transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'input_ids': input_ids, 'attention_mask': attn, 'image': img_tensor, 'label': label}

# -------------------------
# Model
# -------------------------
class LateFusionModel(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', proj_dim=512, hidden_dim=512, num_labels=3, freeze_text=True, freeze_image=False):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        text_dim = self.text_encoder.config.hidden_size

        backbone = models.resnet50(pretrained=True)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        self.proj_text = nn.Linear(text_dim, proj_dim)
        self.proj_image = nn.Linear(feat_dim, proj_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(proj_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels)
        )

        if freeze_text:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        if freeze_image:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, images):
        t_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        t_cls = t_out.last_hidden_state[:,0]
        e_t = self.proj_text(t_cls)

        feats = self.image_encoder(images)
        e_i = self.proj_image(feats)

        fused = torch.cat([e_t, e_i], dim=1)
        logits = self.classifier(fused)
        return logits

# -------------------------
# Training & Evaluation
# -------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc='train'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attn = batch['attention_mask'].to(device)
        imgs = batch['image'].to(device)
        labels = batch['label'].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn, images=imgs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate_model(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='eval'):
            input_ids = batch['input_ids'].to(device)
            attn = batch['attention_mask'].to(device)
            imgs = batch['image'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attn, images=imgs)
            preds += logits.argmax(dim=1).cpu().tolist()
            trues += labels.cpu().tolist()
    
    # 计算指标
    f1_macro = f1_score(trues, preds, average='macro')
    f1_weighted = f1_score(trues, preds, average='weighted')
    acc = accuracy_score(trues, preds)
    report = classification_report(trues, preds, target_names=['positive', 'negative', 'neutral'], digits=4)
    cm = confusion_matrix(trues, preds)
    
    return f1_macro, f1_weighted, acc, report, cm, trues, preds

# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default='cleaned/train_clean.txt')
    parser.add_argument('--val-file', type=str, default='cleaned/val_clean.txt')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--freeze-text', action='store_true')
    parser.add_argument('--freeze-image', action='store_true')
    parser.add_argument('--output-path', type=str, default='best_latefusion.pt')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--output-dir', type=str, default='results/latefusion', help='Directory to save results')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

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

        train_ds = FusionDataset(args.train_file, args.data_dir, train=True)
        val_ds = FusionDataset(args.val_file, args.data_dir, train=False)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # 计算class weights
        df_train = pd.read_csv(args.train_file, sep='\t')
        labels = [LABEL2ID[str(t).strip()] for t in df_train['tag'].tolist()]
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array(list(LABEL2ID.values())), y=np.array(labels))
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
        print(f"Class weights: {class_weights.tolist()}")

        model = LateFusionModel(freeze_text=args.freeze_text, freeze_image=args.freeze_image).to(device)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # 早停机制
        best_f1 = 0
        patience = args.patience
        counter = 0

        # 记录每个epoch的指标
        metrics = []

        for epoch in range(1, args.epochs+1):
            print(f"===== Epoch {epoch} =====")
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
            f1_macro, f1_weighted, acc, report, cm, trues, preds = evaluate_model(model, val_loader, device)
            
            # 打印详细指标
            print(f"Train loss: {train_loss:.4f}")
            print(f"Val macro-F1: {f1_macro:.4f} | Val weighted-F1: {f1_weighted:.4f} | Val acc: {acc:.4f}")
            print("\nClassification Report:")
            print(report)
            print("\nConfusion Matrix:")
            print(cm)
            print()
            
            # 记录指标
            metrics.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_macro_f1': f1_macro,
                'val_weighted_f1': f1_weighted,
                'val_acc': acc
            })
            
            # 保存当前epoch的报告
            epoch_report_file = os.path.join(args.output_dir, f'epoch_{epoch}_report.txt')
            with open(epoch_report_file, 'w', encoding='utf-8') as f:
                f.write(f"Epoch {epoch} Report\n")
                f.write(f"Train loss: {train_loss:.4f}\n")
                f.write(f"Val macro-F1: {f1_macro:.4f} | Val weighted-F1: {f1_weighted:.4f} | Val acc: {acc:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(report + '\n\n')
                f.write("Confusion Matrix:\n")
                f.write(str(cm) + '\n')
            
            # 早停和模型保存
            if f1_macro > best_f1:
                best_f1 = f1_macro
                torch.save(model.state_dict(), args.output_path)
                print(f"Saved best model: {args.output_path}")
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

        # 最终评估
        print("\n===== Final Evaluation =====")
        model.load_state_dict(torch.load(args.output_path, map_location=device))
        f1_macro, f1_weighted, acc, report, cm, trues, preds = evaluate_model(model, val_loader, device)
        print(f"Final val macro-F1: {f1_macro:.4f}")
        print(f"Final val weighted-F1: {f1_weighted:.4f}")
        print(f"Final val acc: {acc:.4f}")
        print("\nFinal Classification Report:")
        print(report)
        print("\nFinal Confusion Matrix:")
        print(cm)
        
        # 保存最终报告
        final_report_file = os.path.join(args.output_dir, 'final_report.txt')
        with open(final_report_file, 'w', encoding='utf-8') as f:
            f.write("Final Evaluation Report\n")
            f.write(f"Best val macro-F1: {best_f1:.4f}\n")
            f.write(f"Final val macro-F1: {f1_macro:.4f}\n")
            f.write(f"Final val weighted-F1: {f1_weighted:.4f}\n")
            f.write(f"Final val acc: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report + '\n\n')
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + '\n')
        print(f"Saved final report to: {final_report_file}")
        
        print(f"\nAll results saved to: {args.output_dir}")
    finally:
        tee.close()
