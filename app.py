import gradio as gr
from transformers import pipeline

classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def predict_news(text):
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]
    if label == "NEGATIVE":
        return f"❌ Fake News ({score:.2f} confidence)"
    else:
        return f"✅ Real News ({score:.2f} confidence)"

gr.Interface(fn=predict_news, inputs="text", outputs="text").launch()
