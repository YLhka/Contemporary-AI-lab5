# train_clipfusion_full.py
"""
CLIP joint-embedding fusion: use CLIP text & image features -> concat -> MLP -> CrossEntropy

Features:
- Train / val / test inference in one script
- Uses CLIPProcessor for preprocessing (text+image)
- Computes class weights, supports freezing CLIP params (--freeze-clip)
- Saves best model and validation report, can output test predictions (replace null in test_without_label.txt)

Usage (train):
python train_clipfusion_full.py \
  --train-file cleaned/train_clean.txt \
  --val-file cleaned/val_clean.txt \
  --test-file test_without_label.txt \
  --data-dir data \
  --clip-model openai/clip-vit-base-patch32 \
  --epochs 4 \
  --batch-size 32 \
  --lr 1e-4 \
  --output best_clipfusion.pt

Usage (inference only):
python train_clipfusion_full.py --predict --checkpoint best_clipfusion.pt --test-file test_without_label.txt --data-dir data --clip-model openai/clip-vit-base-patch32 --batch-size 32
"""
import argparse
import os
from pathlib import Path
import json
import warnings
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 自定义collate_fn来处理不同长度的输入
def collate_fn(batch):
    max_len = max([len(item['input_ids']) for item in batch])
    input_ids = []
    attention_mask = []
    pixel_values = []
    labels = []
    guids = []
    
    for item in batch:
        # 填充input_ids和attention_mask
        pad_len = max_len - len(item['input_ids'])
        input_ids.append(torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
        pixel_values.append(item['pixel_values'])
        if 'label' in item:
            labels.append(item['label'])
        guids.append(item['guid'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'pixel_values': torch.stack(pixel_values),
        'label': torch.tensor(labels) if labels else None,
        'guid': guids
    }

# Grad-CAM实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        # 为目标层注册前向和反向钩子
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        
        # 查找目标层并注册钩子
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                self.hook_handles.append(module.register_backward_hook(backward_hook))
                break
    
    def remove_hooks(self):
        # 移除钩子
        for handle in self.hook_handles:
            handle.remove()
    
    def __call__(self, input_ids, attention_mask, pixel_values, target_class=None):
        # 前向传播
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        
        # 如果没有指定目标类别，使用预测的类别
        if target_class is None:
            target_class = torch.argmax(output, dim=1)
        
        # 反向传播到目标类别
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # 计算加权激活
        cam = torch.sum(weights * self.activations, dim=1)
        # 应用ReLU
        cam = torch.relu(cam)
        # 归一化
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam

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

class CLIPDataset(Dataset):
    """Dataset that uses CLIPProcessor to produce input_ids, attention_mask, pixel_values."""
    def __init__(self, csv_file=None, test_file=None, data_dir='data', processor=None, train=True):
        """
        Provide either csv_file (with guid,text_clean,tag) for train/val
        or test_file (with guid, tag=null) for test prediction.
        """
        self.processor = processor
        self.data_dir = Path(data_dir)
        self.items = []
        if csv_file:
            df = pd.read_csv(csv_file, sep='\t')
            for _, r in df.iterrows():
                guid = str(r['guid'])
                text = r.get('text_clean', '') if 'text_clean' in r.index else r.get('text', '')
                img_path = r.get('img_path', '') if 'img_path' in r.index else str(self.data_dir / f"{guid}.jpg")
                tag = r['tag']
                self.items.append({'guid': guid, 'text': str(text), 'img_path': str(img_path), 'tag': str(tag)})
        elif test_file:
            # 尝试读取测试文件，处理没有列名的情况
            try:
                # 尝试带列名读取
                df = pd.read_csv(test_file, sep='\t')
                if 'guid' in df.columns:
                    # 有列名的情况
                    for _, r in df.iterrows():
                        guid = str(r['guid'])
                        text = r.get('text_clean', '') if 'text_clean' in r.index else r.get('text', '')
                        img_path = r.get('img_path', '') if 'img_path' in r.index else str(self.data_dir / f"{guid}.jpg")
                        self.items.append({'guid': guid, 'text': str(text), 'img_path': str(img_path), 'tag': None})
                else:
                    # 没有列名的情况，假设第一列是guid
                    for _, r in df.iterrows():
                        guid = str(r.iloc[0])
                        text = str(r.iloc[1]) if len(r) > 1 else ''
                        img_path = str(self.data_dir / f"{guid}.jpg")
                        self.items.append({'guid': guid, 'text': text, 'img_path': img_path, 'tag': None})
            except Exception:
                # 完全无法解析的情况，尝试逐行读取
                with open(test_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split('\t')
                        guid = parts[0]
                        text = parts[1] if len(parts) > 1 else ''
                        img_path = str(self.data_dir / f"{guid}.jpg")
                        self.items.append({'guid': guid, 'text': text, 'img_path': img_path, 'tag': None})
        else:
            raise ValueError("Either csv_file or test_file must be provided.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        # 尝试从txt文件中读取文本
        guid = it['guid']
        txt_path = str(self.data_dir / f"{guid}.txt")
        if Path(txt_path).exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            except Exception:
                text = ""
        else:
            text = it['text'] if it['text'] is not None else ""
        
        img_path = it['img_path']
        if img_path and Path(img_path).exists():
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception:
                image = Image.new('RGB', (224,224), (0,0,0))
        else:
            image = Image.new('RGB', (224,224), (0,0,0))

        # processor handles tokenization and image preproc; return_tensors='pt' yields batch dim=1
        proc_out = self.processor(text=text, images=image, return_tensors='pt', padding=True)
        # squeeze batch dim
        input_ids = proc_out['input_ids'].squeeze(0)
        attention_mask = proc_out['attention_mask'].squeeze(0)
        pixel_values = proc_out['pixel_values'].squeeze(0)
        result = {
            'guid': it['guid'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }
        # 只在有标签时添加label键
        if it['tag'] is not None and it['tag'] != 'null':
            label = LABEL2ID[str(it['tag']).strip()]
            result['label'] = label
        return result

class CLIPFusionClassifier(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', proj_dim=512, hidden_dim=512, num_labels=3, freeze_clip=False):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        # try to get a sensible feature dimension: use projection_dim if present, else text hidden size
        cfg = self.clip.config
        if hasattr(cfg, 'projection_dim') and cfg.projection_dim:
            clip_dim = cfg.projection_dim
        else:
            try:
                clip_dim = self.clip.text_model.config.hidden_size
            except Exception:
                clip_dim = 512
        self.proj_text = nn.Linear(clip_dim, proj_dim)
        self.proj_image = nn.Linear(clip_dim, proj_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(proj_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_labels)
        )
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        # Use CLIPModel's get_text_features / get_image_features for pooled features
        # They internally use projection if available
        text_feats = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        image_feats = self.clip.get_image_features(pixel_values=pixel_values)
        e_t = self.proj_text(text_feats)
        e_i = self.proj_image(image_feats)
        fused = torch.cat([e_t, e_i], dim=1)
        logits = self.classifier(fused)
        return logits

    def get_vision_features(self, pixel_values):
        # 获取视觉特征，用于Grad-CAM
        with torch.no_grad():
            # 直接调用vision_model获取特征
            vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
            # 返回最后一层隐藏状态和池化特征
            return vision_outputs.last_hidden_state, vision_outputs.pooler_output

def compute_class_weights(csv_file):
    df = pd.read_csv(csv_file, sep='\t')
    labels = [LABEL2ID[str(x).strip()] for x in df['tag'].tolist()]
    classes = np.array(list(LABEL2ID.values()))
    cw = compute_class_weight(class_weight='balanced', classes=classes, y=np.array(labels))
    return torch.tensor(cw, dtype=torch.float)

def train_one_epoch(model, loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0.0
    preds, trues = [], []
    for batch in tqdm(loader, desc='train'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * labels.size(0)
        preds += logits.argmax(dim=1).detach().cpu().tolist()
        trues += labels.detach().cpu().tolist()
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, preds, trues

def eval_epoch(model, loader, device, loss_fn=None):
    model.eval()
    preds, trues, logits_list, guids, texts, img_paths = [], [], [], [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc='eval'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            if loss_fn is not None:
                total_loss += loss_fn(logits, labels).item() * labels.size(0)
            preds += logits.argmax(dim=1).cpu().tolist()
            trues += labels.cpu().tolist()
            logits_list += logits.cpu().tolist()
            guids += batch.get('guid', [])
            # 由于batch中没有text和img_path，需要在外部处理
    avg_loss = (total_loss / len(loader.dataset)) if loss_fn is not None else None
    macro = f1_score(trues, preds, average='macro')
    weighted = f1_score(trues, preds, average='weighted')
    acc = accuracy_score(trues, preds)
    report = classification_report(trues, preds, target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))], digits=4)
    cm = confusion_matrix(trues, preds)
    return avg_loss, macro, weighted, acc, report, cm, trues, preds, logits_list, guids

def infer_test(model, loader, device):
    model.eval()
    guids = []
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='predict'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            pred = logits.argmax(dim=1).cpu().tolist()
            guids += batch.get('guid', [])
            preds += pred
    return guids, preds

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
        
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        
        # train/val flow
        train_ds = CLIPDataset(csv_file=args.train_file, data_dir=args.data_dir, processor=processor, train=True)
        val_ds = CLIPDataset(csv_file=args.val_file, data_dir=args.data_dir, processor=processor, train=False)
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

        # class weights
        cw = compute_class_weight(class_weight='balanced', classes=np.array(list(LABEL2ID.values())),
                                  y=[LABEL2ID[x.strip()] for x in pd.read_csv(args.train_file, sep='\t')['tag'].tolist()])
        cw = torch.tensor(cw, dtype=torch.float).to(device)
        print(f"Class weights: {cw.tolist()}")

        model = CLIPFusionClassifier(clip_model_name=args.clip_model, proj_dim=args.proj_dim, hidden_dim=args.hidden_dim, freeze_clip=args.freeze_clip).to(device)
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*total_steps), num_training_steps=total_steps)
        loss_fn = nn.CrossEntropyLoss(weight=cw)

        # 早停机制
        best_f1 = -1.0
        patience = args.patience
        counter = 0

        # 记录每个epoch的指标
        metrics = []

        for epoch in range(1, args.epochs+1):
            print(f"===== Epoch {epoch} =====")
            train_loss, train_preds, train_trues = train_one_epoch(model, train_loader, optimizer, scheduler, device, loss_fn)
            val_loss, val_f1, val_f1_weighted, val_acc, report, cm, val_trues, val_preds, _, _ = eval_epoch(model, val_loader, device, loss_fn)
            
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

        # final eval + report
        print("\n===== Final Evaluation =====")
        model.load_state_dict(torch.load(args.output, map_location=device))
        val_loss, val_f1, val_f1_weighted, val_acc, report, cm, val_trues, val_preds, val_logits, val_guids = eval_epoch(model, val_loader, device, loss_fn)
        print(f"Final val macro-F1: {val_f1:.4f} | Final val weighted-F1: {val_f1_weighted:.4f} | Final val acc: {val_acc:.4f}")
        print("\nFinal Classification Report:")
        print(report)
        print("\nFinal Confusion Matrix:")
        print(cm)
        
        # 错误分析：识别分错的样本
        print("\n===== Error Analysis =====")
        misclassified = []
        for i, (true, pred, logits, guid) in enumerate(zip(val_trues, val_preds, val_logits, val_guids)):
            if true != pred:
                # 获取原始样本信息
                sample = val_ds.items[i]
                # 尝试从txt文件中读取文本
                txt_path = str(Path(args.data_dir) / f"{guid}.txt")
                if Path(txt_path).exists():
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                    except Exception:
                        text = ""
                else:
                    text = sample['text'] if sample['text'] is not None else ""
                
                misclassified.append({
                    'guid': guid,
                    'text': text,
                    'img_path': sample['img_path'],
                    'true_label': ID2LABEL[true],
                    'pred_label': ID2LABEL[pred],
                    'logits': logits
                })
        
        print(f"Total misclassified samples: {len(misclassified)}")
        print(f"Error rate: {len(misclassified)/len(val_trues):.4f}")
        
        # 保存分错的样本到文件
        error_analysis_file = os.path.join(args.output_dir, 'error_analysis.json')
        with open(error_analysis_file, 'w', encoding='utf-8') as f:
            json.dump(misclassified, f, ensure_ascii=False, indent=2)
        print(f"Saved error analysis to: {error_analysis_file}")
        
        # 保存最终报告
        final_report_file = os.path.join(args.output_dir, 'final_report.txt')
        with open(final_report_file, 'w', encoding='utf-8') as f:
            f.write("Final Evaluation Report\n")
            f.write(f"Best val macro-F1: {best_f1:.4f}\n")
            f.write(f"Final val macro-F1: {val_f1:.4f} | Final val weighted-F1: {val_f1_weighted:.4f} | Final val acc: {val_acc:.4f}\n\n")

            f.write("Classification Report:\n")
            f.write(report + '\n\n')
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + '\n\n')
            
            f.write("Error Analysis:\n")
            f.write(f"Total misclassified samples: {len(misclassified)}\n")
            f.write(f"Error rate: {len(misclassified)/len(val_trues):.4f}\n")
        print(f"Saved final report to: {final_report_file}")
        
        print(f"\nAll results saved to: {args.output_dir}")
    finally:
        tee.close()

def visualize_bad_cases(error_analysis_file, model_path, processor, device, output_dir, data_dir='code/data'):
    """
    可视化bad-case，包括文本、图片、模型logits
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    from pathlib import Path
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 加载错误分析文件
    with open(error_analysis_file, 'r', encoding='utf-8') as f:
        misclassified = json.load(f)
    
    # 加载模型
    model = CLIPFusionClassifier(freeze_clip=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 可视化前10个bad-case
    for i, case in enumerate(misclassified[:10]):
        guid = case['guid']
        text = case['text']
        # 使用data_dir构造正确的图片路径
        img_path = str(Path(data_dir) / f"{guid}.jpg")
        true_label = case['true_label']
        pred_label = case['pred_label']
        logits = case['logits']
        
        # 加载图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            print(f"无法加载图片 {img_path}")
            continue
        
        # 预处理输入
        inputs = processor(text=text, images=image, return_tensors='pt', padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        pixel_values = inputs['pixel_values'].to(device)
        
        # 获取模型预测
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        
        # 创建可视化
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # 显示原始图片
        ax.imshow(image)
        ax.set_title(f"GUID: {guid}\nPred: {pred_label} | True: {true_label}")
        ax.axis('off')
        
        # 添加文本和logits信息
        plt.figtext(0.5, 0.01, f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}", 
                   ha="center", fontsize=10, wrap=True)
        plt.figtext(0.5, 0.05, f"Logits - Positive: {logits[0]:.4f}, Negative: {logits[1]:.4f}, Neutral: {logits[2]:.4f}", 
                   ha="center", fontsize=10)
        plt.figtext(0.5, 0.09, f"Probabilities - Positive: {probs[0]:.4f}, Negative: {probs[1]:.4f}, Neutral: {probs[2]:.4f}", 
                   ha="center", fontsize=10)
        
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.savefig(output_dir / f"bad_case_{i}_{guid}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"已生成bad-case可视化: {output_dir / f'bad_case_{i}_{guid}.png'}")

if __name__ == '__main__':
    import time
    p = argparse.ArgumentParser()
    p.add_argument('--train-file', type=str, default='cleaned/train_clean.txt')
    p.add_argument('--val-file', type=str, default='cleaned/val_clean.txt')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch32')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--proj-dim', type=int, default=512)
    p.add_argument('--hidden-dim', type=int, default=512)
    p.add_argument('--freeze-clip', action='store_true')
    p.add_argument('--output', type=str, default='best_clipfusion.pt')
    p.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--output-dir', type=str, default='results/clipfusion', help='Directory to save results')
    p.add_argument('--visualize', action='store_true', help='可视化bad-case')
    p.add_argument('--error-analysis-file', type=str, default=None, help='错误分析文件路径')
    args = p.parse_args()
    
    if args.visualize and args.error_analysis_file:
        # 可视化模式
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from transformers import CLIPProcessor
        processor = CLIPProcessor.from_pretrained(args.clip_model)
        model_path = args.output
        visualize_bad_cases(args.error_analysis_file, model_path, processor, device, os.path.join(args.output_dir, 'visualizations'), data_dir=args.data_dir)
    else:
        # 训练模式
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        main(args)
