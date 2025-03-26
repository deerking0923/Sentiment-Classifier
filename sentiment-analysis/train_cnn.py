import sys, os
os.makedirs("results", exist_ok=True)

# 터미널 + 파일 로그 출력용 Logger 클래스
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("results/log_cnn.txt")
sys.stderr = sys.stdout

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
import json
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

# 간단한 CNN 분류기
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# 데이터 로딩
raw_dataset = load_dataset("csv", data_files={"train": "data/ratings_train.txt", "test": "data/ratings_test.txt"}, sep="\t")
raw_dataset = raw_dataset.rename_column("label", "labels")

# 학습 데이터 일부만 사용 (10000개)
raw_dataset["train"] = raw_dataset["train"].shuffle(seed=42).select(range(10000))
raw_dataset["test"] = raw_dataset["test"].shuffle(seed=42).select(range(10000))

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", cache_dir="./hf_cache", trust_remote_code=True)

# 전처리
max_len = 128
def preprocess(example):
    tokens = tokenizer(
        text=str(example["document"]),
        padding="max_length",
        truncation=True,
        max_length=max_len
    )
    return {"input_ids": tokens["input_ids"], "labels": example["labels"]}

dataset = raw_dataset.map(preprocess)
dataset.set_format("torch", columns=["input_ids", "labels"])

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=64)

# 모델, 손실함수, 옵티마이저
vocab_size = tokenizer.vocab_size
model = CNNClassifier(vocab_size=vocab_size, embed_dim=128, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 루프
for epoch in range(3):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):
        inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}] Step {i+1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

    print(f"[Epoch {epoch+1}] Average Loss: {total_loss / len(train_loader):.4f}")

# 평가
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

with open("results/cnn_result.json", "w") as f:
    json.dump({"model": "CNN", "accuracy": acc, "f1": f1}, f, indent=4)

print("[CNN] Accuracy:", acc, "F1:", f1)
