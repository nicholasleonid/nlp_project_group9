import re

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_PATHS = {
    "T5": "nikiloves/indonesian-t5-summarizer",
    "BART": "nikiloves/indonesian-bart-summarizer",
    "PEGASUS": "nikiloves/indonesian-pegasus-summarizer",
    "mT5": "nikiloves/indonesian-mt5-summarizer",
}


st.set_page_config(page_title="NLP Summarizer", layout="centered")


@st.cache_resource
def load_model(model_name):
    path = MODEL_PATHS[model_name]

    if model_name in ["BART", "PEGASUS"]:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    model.eval()

    return tokenizer, model


def clean_summary(summary):
    summary = re.sub(r"<extra_id_\d+>", "", summary)
    summary = summary.replace("▁", " ")
    summary = summary.replace("_", " ")
    summary = re.sub(r"\s+", " ", summary)

    return summary.strip()


def generate_summary(text, model_name):
    tokenizer, model = load_model(model_name)

    if model_name in ["T5", "mT5"]:
        text = "summarize: " + text

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    generation_settings = {
        "max_length": 90,
        "min_length": 25,
        "num_beams": 5,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
        "early_stopping": True,
    }

    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            **generation_settings,
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return clean_summary(summary)


st.title("Indonesian News Summarization")
st.write("Choose a model and generate a summary.")

model_choice = st.selectbox(
    "Choose model",
    ["T5", "BART", "PEGASUS", "mT5"],
)

text_input = st.text_area(
    "Paste Indonesian article text here",
    height=250,
)

if st.button("Generate Summary"):
    if not text_input.strip():
        st.warning("Please enter text first.")
    else:
        with st.spinner("Generating summary..."):
            summary = generate_summary(text_input, model_choice)

        st.subheader("Summary")
        st.write(summary)