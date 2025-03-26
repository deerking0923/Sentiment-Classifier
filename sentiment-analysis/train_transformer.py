import sys, os
os.makedirs("results", exist_ok=True)

# 터미널 + 파일 로그 출력용 Logger 클래스
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message)
        except UnicodeEncodeError:
            self.log.write(message.encode("utf-8", "ignore").decode("utf-8"))

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("results/log_transformer.txt")
sys.stderr = sys.stdout

import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# 데이터 로딩
raw_dataset = load_dataset("csv", data_files={"train": "data/ratings_train.txt", "test": "data/ratings_test.txt"}, sep="\t")
raw_dataset = raw_dataset.rename_column("label", "labels")

# 학습 데이터 일부만 사용 (10000개)
raw_dataset["train"] = raw_dataset["train"].shuffle(seed=42).select(range(10000))
raw_dataset["test"] = raw_dataset["test"].shuffle(seed=42).select(range(10000))

# 토크나이저
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", cache_dir="./hf_cache", trust_remote_code=True)

# 전처리 함수
def preprocess(example):
    tokens = tokenizer(
        text=str(example["document"]),
        padding="max_length",
        truncation=True,
        max_length=128
    )
    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "labels": example["labels"]
    }

# 전처리 적용
encoded_dataset = raw_dataset.map(preprocess)
encoded_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 모델 로딩
model = AutoModelForSequenceClassification.from_pretrained(
    "monologg/kobert",
    cache_dir="./hf_cache",
    num_labels=2,
    trust_remote_code=True
)

# 평가 함수 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

# Trainer 설정
training_args = TrainingArguments(
    output_dir="results/transformer_output",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    report_to=[],
    fp16=True
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    compute_metrics=compute_metrics
)

# 학습 및 평가
trainer.train()
eval_result = trainer.evaluate()

# 결과 저장
with open("results/kobert_result.json", "w", encoding="utf-8") as f:
    json.dump({"model": "KoBERT", "accuracy": eval_result["eval_accuracy"], "f1": eval_result["eval_f1"]}, f, indent=4, ensure_ascii=False)

print("[KoBERT] Accuracy:", eval_result["eval_accuracy"], "F1:", eval_result["eval_f1"])
