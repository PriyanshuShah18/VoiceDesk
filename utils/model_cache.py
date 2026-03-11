import streamlit as st
from transformers import AutoModel, VitsModel, AutoTokenizer

@st.cache_resource
def load_indic_conformer():

    model = AutoModel.from_pretrained(
        "ai4bharat/indic-conformer-600m-multilingual",
        trust_remote_code=True,
    )
    return model

@st.cache_resource
def load_mms_tts(model_id: str):

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
    )

    model = VitsModel.from_pretrained(
        model_id,
        low_cpu_mem_usage=True,
        torch_dtype="auto",
    )

    return tokenizer,model