from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from models.transformer import load_kobert_model
import evaluate
import json
import os

import sys, os
os.makedirs("results", exist_ok=True)
sys.stdout = open("results/log_transformer.txt", "w")
sys.stderr = sys.stdout

# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("monologg/kobert", trust_remote_code=True)
model = load_kobert_model()

# 전처리 함수
def preprocess(batch):
    texts = [str(x) for x in batch["document"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=128)

# 데이터셋 불러오기
dataset = load_dataset(
    "csv",
    data_files={"train": "data/ratings_train.txt", "test": "data/ratings_test.txt"},
    sep="\t"
)
dataset = dataset.rename_column("label", "labels")
dataset = dataset.map(preprocess, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 평가 지표
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels)

# 학습 파라미터
training_args = TrainingArguments(
    output_dir="outputs",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_total_limit=1,
    report_to="none"
)

# Trainer 세팅
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)

# 학습
trainer.train()

results = trainer.evaluate()

# 결과 저장
os.makedirs("results", exist_ok=True)
result = {
    "model": "KoBERT",
    "accuracy": results["eval_accuracy"],
    "f1": results.get("eval_f1", None)
}
with open("results/kobert_result.json", "w") as f:
    json.dump(result, f, indent=4)

print("결과 저장 완료:", result)
