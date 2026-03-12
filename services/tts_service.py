import logging
import torch
import os
from uuid import uuid4

from transformers import VitsModel, AutoTokenizer
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from utils.model_cache import load_mms_tts

import tempfile

import numpy as np

import soundfile as sf  # Faster audio writing

import streamlit as st

class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    text: str = Field(description="The text to synthesize")
    language_code: str = Field(default="en", description="The ISO code of desired output language")
    
    @model_validator(mode="after")
    def validate_script_safety(self):
        # Prevent passing English text to non-English models
        latin_chars = sum(1 for c in self.text if 'a' <= c.lower() <= 'z')
        total_chars = sum(1 for c in self.text if not c.isspace()) or 1
        latin_ratio = latin_chars / total_chars
        
        if latin_ratio > 0.5 and self.language_code in ["gu", "hi"]:
            logging.warning(f"Text appears to be English (Latin script, ratio: {latin_ratio:.2f}) but requested language is '{self.language_code}'. Falling back to 'en' model.")
            self.language_code = "en"
            
        return self

@st.cache_resource(show_spinner=False)
def cached_load_mms(model_id):
    tokenizer, model = load_mms_tts(model_id)
    return tokenizer, model

class TTSService:
    def __init__(self):
        """
        Initializes the TTS service.
        """
        logging.info("Initializing TTS Service with local Meta MMS fallback")
        
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        #self.output_dir = output_dir
        #os.makedirs(self.output_dir, exist_ok=True)
        self.device = torch.device("cpu")

        torch.set_grad_enabled(False)

        torch.set_num_threads(2)

        # Temp directory 
        self.output_dir= os.path.join(tempfile.gettempdir(), "tts_audio")
        os.makedirs(self.output_dir, exist_ok=True)
        # Model mapping for local fallback
        self.model_map = {
            "gu": "facebook/mms-tts-guj",
            #"hi": "facebook/mms-tts-hin",
            "en": "facebook/mms-tts-eng"
        }
        
        # Cache for loaded models
        self.loaded_models = {}
        self.loaded_tokenizers = {}

    def _get_mms_model(self, lang_code):
        """
        Lazily loads the requested facebook model.
        """
        if lang_code not in self.model_map:
            logging.warning(f"No local TTS model for language '{lang_code}', falling back to English.")
            lang_code = "en"
            
        if lang_code not in self.loaded_models:

            model_id = self.model_map[lang_code]

            logging.info(f"Loading local TTS model: {model_id}")
            
            #tokenizer,model = load_mms_tts(model_id)
            tokenizer, model = cached_load_mms(model_id)
    
            model = model.to(self.device, non_blocking=True)
            model.eval()

            # PyTorch model compilation for faster inference
            #model= torch.compile(model)

            self.loaded_models[lang_code] = model
            self.loaded_tokenizers[lang_code] = tokenizer
            

        return self.loaded_tokenizers[lang_code], self.loaded_models[lang_code]
    

    def generate_speech(self, text: str,language_code="en") -> str:
        """
        Generates speech from text and saves it to a WAV file using local Meta MMS.
        """
        max_chars = 400

        if len(text) > max_chars:
            text = text[:max_chars]
            text = text[:text.rfind(" ")] + "..."

        request = SpeechRequest(text=text, language_code= language_code)

        lang = request.language_code

        tokenizer, model = self._get_mms_model(lang)

        try:
            logging.info(f"Generating speech for language '{lang}'")

            inputs = tokenizer(request.text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k,v in inputs.items()}

            with torch.no_grad():
                output= model(**inputs)

            waveform = output.waveform

            audio = waveform.squeeze().cpu().numpy()

            # Squeeze() -> Removes batch dimension
            # cpu() -> Ensures tensor on CPU
            # numpy() -> Convert to numpy, Results in an array of audio samples
            
            
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak
            #audio = output.cpu().numpy().squeeze()

            # Normalize audio

            #audio = audio / np.max(np.abs(audio))

            sample_rate = model.config.sampling_rate

            filename = f"response_{uuid4().hex[:8]}.wav"
            output_path = os.path.join(self.output_dir, filename)

            # Faster audio writing.
            sf.write(output_path, audio, sample_rate, format="WAV")

            logging.info(f"TTS audio saved to {output_path}")

            return output_path

        except Exception as e:

            logging.error(f"TTS generation failed: {e}")

            return None
