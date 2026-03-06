import os 
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key,default=None):
    if key in st.secrets:
        return st.secrets[key]
    return os.getenv(key,default)
    