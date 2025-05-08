import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
from datasets import load_dataset






# モデルとトークナイザーのロード
model_path = "./scripts/language-bert-model"  # 保存したディレクトリ名に変更

model = AutoModelForSequenceClassification.from_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# ラベル名
dataset = load_dataset("code-search-net/code_search_net",trust_remote_code=True)
dataset = dataset.class_encode_column("language")
labels = dataset["train"].features["language"].names

print(labels)

# 推論関数
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = torch.argmax(logits, dim=1).item()
    return labels[pred]

# Gradioインターフェース
demo = gr.Interface(fn=classify_text, inputs="text", outputs="text", title="言語分類BERT")

demo.launch(server_port=8888)
