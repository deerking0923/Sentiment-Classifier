# compare.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import sys

# 로그 파일 저장 (stdout 리디렉션)
os.makedirs("results", exist_ok=True)
sys.stdout = open("results/log_compare.txt", "w")
sys.stderr = sys.stdout

# 결과 디렉토리에서 모든 json 파일 로드
results_dir = "results"
result_files = [f for f in os.listdir(results_dir) if f.endswith("_result.json")]

records = []
for filename in result_files:
    with open(os.path.join(results_dir, filename), "r") as f:
        record = json.load(f)
        records.append({
            "Model": record["model"],
            "Accuracy": record["accuracy"],
            "F1": record["f1"]
        })

# DataFrame 생성 및 출력 (소수점 3자리)
df = pd.DataFrame(records)
df[["Accuracy", "F1"]] = df[["Accuracy", "F1"]].round(3)
print("\n===== 모델 비교 결과표 =====\n")
print(df)

# 정확도 시각화
plt.figure(figsize=(8, 5))
bars = plt.bar(df["Model"], df["Accuracy"], color="skyblue")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.5, 1.0)
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
plt.savefig("results/accuracy_plot.png")
plt.show()

# F1 시각화
plt.figure(figsize=(8, 5))
bars = plt.bar(df["Model"], df["F1"], color="lightcoral")
plt.title("Model F1 Score Comparison")
plt.ylabel("F1 Score")
plt.ylim(0.5, 1.0)
plt.grid(True, axis="y", linestyle="--", alpha=0.7)
plt.bar_label(bars, fmt="%.3f", padding=3)
plt.tight_layout()
plt.savefig("results/f1_plot.png")
plt.show()