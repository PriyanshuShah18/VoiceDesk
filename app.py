import streamlit as st
import os

# Inject Streamlit Cloud secrets into environment (secrets are not available via .env on Cloud)
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # Running locally — .env will be used instead

# Hugging Face cache — use /tmp on Linux/Cloud, keep relative path only on Windows locally
_is_linux = os.name != 'nt'
_hf_cache = "/tmp/huggingface" if _is_linux else "tmp/huggingface"
os.environ["HF_HOME"] = _hf_cache
os.environ["TRANSFORMERS_CACHE"] = _hf_cache
os.environ["HF_DATASETS_CACHE"] = _hf_cache
os.makedirs(_hf_cache, exist_ok=True)

# Force CPU mode for all ML libraries
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CT2_USE_CUDA"] = "0"

import tempfile
import logging
from agent.voice_agent import VoiceAgent

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# Page Config
st.set_page_config(page_title="AI Receptionist", page_icon="", layout="centered")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0c0c1e;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4f46e5;
        color: white;
        font-weight: bold;
    }
    .entity-card {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
    }
    .entity-label {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-bottom: 5px;
    }
    .entity-value {
        font-size: 1.1rem;
        font-weight: bold;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    return VoiceAgent()

# [REMOVED] load_asr() was dead code referencing an undefined IndicConformer class

def main():
    st.write("Converse in Gujarati or English to book an appointment.")
    
    # Language Selection
    st.sidebar.subheader("Language Settings")
    lang_mode = st.sidebar.radio("Detection Mode", ["Auto-Detect", "Gujarati", "Hindi", "English"])
    
    # Provider Selection (Hardcoded to Smart Auto for simplicity)
    selected_provider = "smart"
    
    lang_map = {
        "Auto-Detect": None,
        "Gujarati": "gu",
        "Hindi": "hi",
        "English": "en"
    }
    target_lang = lang_map[lang_mode]

    if "agent" not in st.session_state:
        st.session_state.agent = load_agent()
    
    agent = st.session_state.agent

    # Audio input
    audio_file = st.file_uploader("Upload an audio file (.wav, .mp3, etc.)", type=["wav", "mp3", "m4a", "ogg"])

    if audio_file is not None:
        with st.spinner("Processing request..."):
            # Save the uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_path = tmp_file.name

            try:
                # Run through the standardized pipeline
                result = agent.process_audio(tmp_path, forced_language=target_lang)
                
                # 2. Transcription and Intent
                st.subheader("Results")
                st.info(f"**Transcription ({result['language']}):**\n\n{result['transcription']}")
                
                if "intent" in result:
                    st.caption(f"Detected Intent: {result['intent']}")

                st.divider()
                
                # 3. Entity Cards in a Single Row
                st.subheader("Extracted Details")

                entities = result['entities']
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    val = entities.get("name") or "---"
                    opacity = 1.0 if entities.get("name") else 0.3
                    st.markdown(f'<div class="entity-card" style="opacity: {opacity};"><div class="entity-label">NAME</div><div class="entity-value">{val}</div></div>', unsafe_allow_html=True)
                with col2:
                    val = entities.get("phone") or "---"
                    opacity = 1.0 if entities.get("phone") else 0.3
                    st.markdown(f'<div class="entity-card" style="opacity: {opacity};"><div class="entity-label">PHONE</div><div class="entity-value">{val}</div></div>', unsafe_allow_html=True)
                with col3:
                    val = entities.get("date") or "---"
                    opacity = 1.0 if entities.get("date") else 0.3
                    st.markdown(f'<div class="entity-card" style="opacity: {opacity};"><div class="entity-label">DATE</div><div class="entity-value">{val}</div></div>', unsafe_allow_html=True)
                with col4:
                    val = entities.get("time") or "---"
                    opacity = 1.0 if entities.get("time") else 0.3
                    st.markdown(f'<div class="entity-card" style="opacity: {opacity};"><div class="entity-label">TIME</div><div class="entity-value">{val}</div></div>', unsafe_allow_html=True)

                # Show conversation state
                st.divider()
                st.subheader("Conversation State")

                state = agent.dialogue_manager.get_state()

                st.json(state)

                st.divider()
                st.subheader("Assistant Response")
                #st.write(result['response_text'])
                
                if result['audio_path'] and os.path.exists(result['audio_path']):
                    st.audio(result['audio_path'], format="audio/wav", autoplay=True)
                else:
                    st.error("Audio generation failed.")
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
                logging.exception("Error in Streamlit app")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.sidebar.title("About")
    st.sidebar.info("This is a voice-powered Receptionist")
    if st.sidebar.button("Reset Conversation"):
        if "agent" in st.session_state:
            st.session_state.agent.dialogue_manager.reset_state()
        st.sidebar.success("State Reset!")

if __name__ == "__main__":
    main()
