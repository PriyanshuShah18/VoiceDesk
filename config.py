import os 
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def get_secret(key,default=None):
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return os.getenv(key,default)
        
        
