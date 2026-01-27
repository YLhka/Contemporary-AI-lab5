# predict_test.py
"""
测试集预测脚本
功能：读取 test_without_label.txt，使用训练好的模型进行预测，将 null 替换为预测标签
"""
import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
import pandas as pd
from PIL import Image
from tqdm import tqdm

# 固定随机种子
import random
import numpy as np

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

# 自定义collate_fn来处理不同长度的输入
def collate_fn(batch):
    max_len = max([len(item['input_ids']) for item in batch])
    input_ids = []
    attention_mask = []
    pixel_values = []
    guids = []
    
    for item in batch:
        # 填充input_ids和attention_mask
        pad_len = max_len - len(item['input_ids'])
        input_ids.append(torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)]))
        attention_mask.append(torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)]))
        pixel_values.append(item['pixel_values'])
        guids.append(item['guid'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'pixel_values': torch.stack(pixel_values),
        'guid': guids
    }

class CLIPDataset(Dataset):
    """Dataset for test prediction"""
    def __init__(self, test_file, data_dir='data', processor=None):
        self.processor = processor
        self.data_dir = Path(data_dir)
        self.items = []
        
        # 读取测试文件
        try:
            df = pd.read_csv(test_file, sep=',')
            if 'guid' in df.columns:
                # 有列名的情况
                for _, r in df.iterrows():
                    # 处理guid，确保转换为整数
                    guid_val = r['guid']
                    if isinstance(guid_val, float) and guid_val.is_integer():
                        guid = str(int(guid_val))
                    else:
                        guid = str(guid_val).strip()
                    text = r.get('text_clean', '') if 'text_clean' in r.index else r.get('text', '')
                    img_path = r.get('img_path', '') if 'img_path' in r.index else str(self.data_dir / f"{guid}.jpg")
                    self.items.append({'guid': guid, 'text': str(text), 'img_path': str(img_path)})
            else:
                # 没有列名的情况，假设第一列是guid
                for _, r in df.iterrows():
                    # 处理guid，确保转换为整数
                    guid_val = r.iloc[0]
                    if isinstance(guid_val, float) and guid_val.is_integer():
                        guid = str(int(guid_val))
                    else:
                        guid = str(guid_val).strip()
                    text = str(r.iloc[1]) if len(r) > 1 else ''
                    img_path = str(self.data_dir / f"{guid}.jpg")
                    self.items.append({'guid': guid, 'text': text, 'img_path': img_path})
        except Exception as e:
            print(f"Error reading test file: {e}")
            # 完全无法解析的情况，尝试逐行读取
            with open(test_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"Total lines in test file: {len(lines)}")
                for i, line in enumerate(lines[1:10]):  # 只打印前10行数据
                    line = line.strip()
                    if not line or line.startswith('guid'):
                        continue
                    parts = line.split(',')
                    # 处理guid，确保转换为整数
                    guid_val = parts[0]
                    try:
                        if '.' in guid_val:
                            guid = str(int(float(guid_val)))
                        else:
                            guid = str(int(guid_val))
                    except:
                        guid = guid_val.strip()
                    text = parts[1] if len(parts) > 1 else ''
                    img_path = str(self.data_dir / f"{guid}.jpg")
                    print(f"Line {i+1}: guid={guid}, text={text}, img_path={img_path}")
                    self.items.append({'guid': guid, 'text': text, 'img_path': img_path})
                # 处理剩余行
                for line in lines[10:]:
                    line = line.strip()
                    if not line or line.startswith('guid'):
                        continue
                    parts = line.split(',')
                    # 处理guid，确保转换为整数
                    guid_val = parts[0]
                    try:
                        if '.' in guid_val:
                            guid = str(int(float(guid_val)))
                        else:
                            guid = str(int(guid_val))
                    except:
                        guid = guid_val.strip()
                    text = parts[1] if len(parts) > 1 else ''
                    img_path = str(self.data_dir / f"{guid}.jpg")
                    self.items.append({'guid': guid, 'text': text, 'img_path': img_path})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        # 尝试从txt文件中读取文本
        guid = it['guid']
        txt_path = str(self.data_dir / f"{guid}.txt")
        
        # 添加调试信息，打印前5个样本的处理情况
        if idx < 5:
            print(f"Processing sample {idx}: guid={guid}, txt_path={txt_path}")
        
        if Path(txt_path).exists():
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                if idx < 5:
                    print(f"Sample {idx} text: '{text[:50]}...'" if len(text) > 50 else f"Sample {idx} text: '{text}'")
            except Exception as e:
                print(f"Error reading {txt_path}: {e}")
                text = ""
        else:
            text = it['text'] if it['text'] is not None else ""
            if idx < 5:
                print(f"Sample {idx} using fallback text: '{text}'")
        
        # 确保文本不为空
        if not text:
            text = "This is a default text"
            if idx < 5:
                print(f"Sample {idx} using default text")
        
        img_path = it['img_path']
        if img_path and Path(img_path).exists():
            try:
                image = Image.open(img_path).convert('RGB')
                if idx < 5:
                    print(f"Sample {idx} loaded image: {img_path}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                image = Image.new('RGB', (224,224), (0,0,0))
                if idx < 5:
                    print(f"Sample {idx} using default image")
        else:
            image = Image.new('RGB', (224,224), (0,0,0))
            if idx < 5:
                print(f"Sample {idx} using default image")

        # processor handles tokenization and image preproc; return_tensors='pt' yields batch dim=1
        proc_out = self.processor(text=text, images=image, return_tensors='pt', padding=True)
        # squeeze batch dim
        input_ids = proc_out['input_ids'].squeeze(0)
        attention_mask = proc_out['attention_mask'].squeeze(0)
        pixel_values = proc_out['pixel_values'].squeeze(0)
        
        # 添加调试信息，打印前5个样本的输入张量形状
        if idx < 5:
            print(f"Sample {idx} input_ids shape: {input_ids.shape}")
            print(f"Sample {idx} attention_mask shape: {attention_mask.shape}")
            print(f"Sample {idx} pixel_values shape: {pixel_values.shape}")

        result = {
            'guid': it['guid'],
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values
        }
        return result

class CLIPFusionClassifier(nn.Module):
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', proj_dim=512, hidden_dim=512):
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
            nn.Linear(hidden_dim, 3)
        )

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

def infer_test(model, loader, device):
    model.eval()
    guids = []
    preds = []
    logits_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc='predict'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
            # 添加调试信息，打印第一个样本的logits
            if len(logits_list) == 0:
                print(f"Sample logits: {logits[0].cpu().tolist()}")
                print(f"Sample input_ids shape: {input_ids.shape}")
                print(f"Sample attention_mask shape: {attention_mask.shape}")
                print(f"Sample pixel_values shape: {pixel_values.shape}")
            pred = logits.argmax(dim=1).cpu().tolist()
            guids += batch.get('guid', [])
            preds += pred
            logits_list.append(logits.cpu().tolist())
    # 打印logits统计信息
    all_logits = torch.tensor([item for sublist in logits_list for item in sublist])
    print(f"Logits mean: {all_logits.mean(dim=0).tolist()}")
    print(f"Logits std: {all_logits.std(dim=0).tolist()}")
    return guids, preds

def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载处理器
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    
    # 加载测试数据集
    test_ds = CLIPDataset(test_file=args.test_file, data_dir=args.data_dir, processor=processor)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    print(f"Test dataset size: {len(test_ds)}")
    
    # 加载模型
    model = CLIPFusionClassifier(clip_model_name=args.clip_model, proj_dim=args.proj_dim, hidden_dim=args.hidden_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded model from: {args.checkpoint}")
    
    # 进行预测
    guids, preds = infer_test(model, test_loader, device)
    
    # 将预测结果转换为标签
    pred_labels = [ID2LABEL[pred] for pred in preds]
    
    # 创建预测结果字典
    pred_dict = dict(zip(guids, pred_labels))
    
    # 读取原始测试文件并替换null值
    with open(args.test_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理表头
    header = lines[0].strip()
    output_lines = [header]
    
    # 处理数据行
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        guid = parts[0]
        # 替换null为预测标签
        if guid in pred_dict:
            parts[1] = pred_dict[guid]
        else:
            parts[1] = 'neutral'  # 默认标签
        output_lines.append(','.join(parts))
    
    # 生成输出文件路径
    output_file = os.path.join(args.output_dir, 'test_predictions.txt')
    
    # 写入预测结果
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    print(f"\n=== Prediction Complete ===")
    print(f"Predictions saved to: {output_file}")
    print(f"Total predictions: {len(pred_dict)}")
    
    # 统计预测标签分布
    from collections import Counter
    label_counts = Counter(pred_labels)
    print(f"\nPrediction distribution:")
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/len(pred_labels)*100:.2f}%)")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test-file', type=str, default='test_without_label.txt', help='Test file with null labels')
    p.add_argument('--data-dir', type=str, default='data', help='Directory containing images and text files')
    p.add_argument('--clip-model', type=str, default='openai/clip-vit-base-patch32', help='CLIP model name')
    p.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint file')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size for prediction')
    p.add_argument('--proj-dim', type=int, default=512, help='Projection dimension')
    p.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension')
    p.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    p.add_argument('--output-dir', type=str, default='predictions', help='Directory to save predictions')
    args = p.parse_args()
    
    # 验证文件存在性
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        exit(1)
    if not os.path.exists(args.test_file):
        print(f"Error: Test file not found: {args.test_file}")
        exit(1)
    
    main(args)
