# train_mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from models.mlp import MLPClassifier
import json
import os
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

import sys, os
os.makedirs("results", exist_ok=True)
sys.stdout = open("results/log_mlp.txt", "w")  # train_mlp.py일 경우
sys.stderr = sys.stdout

# 데이터 로딩
raw_dataset = load_dataset("csv", data_files={"train": "data/ratings_train.txt", "test": "data/ratings_test.txt"}, sep="\t")
raw_dataset = raw_dataset.rename_column("label", "labels")

# 토크나이저 (임베딩 없이 단순 임베딩 흉내용)
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")

# 전처리
max_len = 128
def preprocess(example):
    tokens = tokenizer(example["document"], padding="max_length", truncation=True, max_length=max_len)
    return {"input_ids": tokens["input_ids"], "labels": example["labels"]}

dataset = raw_dataset.map(preprocess)
dataset.set_format("torch", columns=["input_ids", "labels"])

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=64)

# 모델, 손실함수, 옵티마이저
model = MLPClassifier(input_dim=max_len, hidden_dim=256)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 루프
for epoch in range(3):
    model.train()
    for batch in train_loader:
        inputs, labels = batch["input_ids"].float().to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 평가
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch["input_ids"].float().to(device), batch["labels"].to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

os.makedirs("results", exist_ok=True)
with open("results/mlp_result.json", "w") as f:
    json.dump({"model": "MLP", "accuracy": acc, "f1": f1}, f, indent=4)

print("[MLP] Accuracy:", acc, "F1:", f1)