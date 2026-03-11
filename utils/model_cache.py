import os
import streamlit as st
from transformers import AutoModel, VitsModel, AutoTokenizer
import logging

# Resolve HF cache: respect any caller-set env var, otherwise use /tmp (works on Linux/Cloud)
_HF_CACHE = os.environ.get("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HF_HOME", _HF_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", _HF_CACHE)
os.makedirs(_HF_CACHE, exist_ok=True)


@st.cache_resource
def load_indic_conformer(token: str = None):
    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        token=token,
        trust_remote_code=True,
    )
    return model


@st.cache_resource
def load_mms_tts(model_id: str):
    hf_cache = os.environ.get("HF_HOME", "/tmp/huggingface")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=hf_cache
    )

    model = VitsModel.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
        cache_dir=hf_cache
    )
    model.eval()

    return tokenizer, model