from transformers import AutoTokenizer
from models.transformer import load_kobert_model
import torch

model = load_kobert_model()
model.load_state_dict(torch.load("outputs/pytorch_model.bin"))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("monologg/kobert")
text = "이 영화 정말 재밌다"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    print("결과:", "긍정" if prediction == 1 else "부정")