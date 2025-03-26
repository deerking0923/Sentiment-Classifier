# train_cnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from models.cnn import CNNClassifier
import json
import os
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score

import sys, os
os.makedirs("results", exist_ok=True)
sys.stdout = open("results/log_cnn.txt", "w")
sys.stderr = sys.stdout

# 데이터 로딩
raw_dataset = load_dataset("csv", data_files={"train": "data/ratings_train.txt", "test": "data/ratings_test.txt"}, sep="\t")
raw_dataset = raw_dataset.rename_column("label", "labels")

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")

max_len = 128
vocab_size = tokenizer.vocab_size

def preprocess(example):
    tokens = tokenizer(example["document"], padding="max_length", truncation=True, max_length=max_len)
    return {"input_ids": tokens["input_ids"], "labels": example["labels"]}

dataset = raw_dataset.map(preprocess)
dataset.set_format("torch", columns=["input_ids", "labels"])

train_loader = torch.utils.data.DataLoader(dataset["train"], batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset["test"], batch_size=64)

model = CNNClassifier(vocab_size=vocab_size, embed_dim=128)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(3):
    model.train()
    for batch in train_loader:
        inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

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

os.makedirs("results", exist_ok=True)
with open("results/cnn_result.json", "w") as f:
    json.dump({"model": "CNN", "accuracy": acc, "f1": f1}, f, indent=4)

print("[CNN] Accuracy:", acc, "F1:", f1)