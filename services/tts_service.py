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

import re
from num2words import num2words
from datetime import datetime



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
    tokenizer, model = load_mms_tts(model_id)  # MMS - Massively Multilingual Speech.
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
            "hi": "facebook/mms-tts-hin",
            "en": "facebook/mms-tts-eng"
        }
        
        # Cache for loaded models
        self.loaded_models = {}
        self.loaded_tokenizers = {}

    @staticmethod
    def crossfade_audio(chunks, fade_samples=1200):
        """
        Smoothly joins audio chunks using crossfade.
        """

        if not chunks:
            return np.array([])
        
        output = chunks[0]

        for chunk in chunks[1:]:

            fade = min(fade_samples, len(output), len(chunk))

            # fade-out previous audio
            fade_out = np.linspace(1, 0, fade)

            # fade-in next audio
            fade_in = np.linspace(0, 1, fade)

            output[-fade:] = output[-fade:] * fade_out + chunk[:fade] * fade_in
            
            output = np.concatenate([output, chunk[fade:]])

        return output

    @staticmethod
    def conversational_chunks(text: str) -> str:
        text = re.sub(r',',', ', text)

        #text = re.sub(r' and ', '... and ', text)
        #text = re.sub(r' but ', '... but ',text)
        #text = re.sub(r' so ','... so ', text)
        #text = re.sub(r' because ', '... because ', text)

        # break very long sentences
        if len(text) > 150:
            text = text.replace(',', '...')

        return text

    @staticmethod
    def add_speech_pauses(text: str) -> str:
        """
        Adds natural pauses for punctuations
        """
        text = re.sub(r',', ', ', text)
        text = re.sub(r'\.','... ',text)
        text = re.sub(r'\?','?... ',text)
        text = re.sub(r'!','!... ',text)

        return text

    @staticmethod
    def normalize_speech_text(text: str) -> str:
        try:
            # Time
            def replace_time(match):
                hour = int(match.group(1))
                meridiem = match.group(2).lower()

                hour_word = num2words(hour)

                if meridiem == "pm":
                    meridiem_word = "pee em"
                else:
                    meridiem_word = "ay em"

                return f"{hour_word} {meridiem_word}"

            text = re.sub(
                r"\b(\d{1,2})\s*(am|pm)\b",
                replace_time,
                text,
                flags=re.IGNORECASE,
            )

            # TIME FORMAT 17:30 -> five thirty
            def replace_colon_time(match):
                hour = int(match.group(1))
                minute = int(match.group(2))

                hour_word = num2words(hour)

                if minute == 0:
                    return f"{hour_word} o clock"

                minute_word = num2words(minute)

                return f"{hour_word} {minute_word}"

            text = re.sub(r"\b(\d{1,2}):(\d{2})\b", replace_colon_time, text)

            # Standalone numbers

            def replace_numbers(match):
                num_str = match.group()

                # If long number -> treat as contact no
                if len(num_str) >= 7:
                    return " ".join(num2words(int(d)) for d in num_str)
                
                # Otherwise normal number
                return num2words(int(num_str))

            text = re.sub(r"\b\d+\b", replace_numbers, text)

        except Exception as e:
            logging.warning(f"Number normalization failed: {e}")
        
        # Dates
        def replace_date(match):
            try:
                dt = datetime.strptime(match.group(), "%d/%m/%Y")
                return dt.strftime("%d %B %Y")
            except:
                return match.group()
            
        text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', replace_date, text)

        # Abbreviations

        abbreviations = {
            "Dr." : "Doctor",
            "Mr." : "Mister",
            "Mrs." : "Missus",
            "Ms." : "Miss",
            "Prof." : "Professor",
            "Inc." : "Incorporated",
            "Ltd." : "Limited",
            "St." : "Street",
            "Ave.": "Avenue"
        }

        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        return text

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

            last_space = text.rfind(" ")
            if last_space != -1:
                text = text[:last_space]
            
            text += "..."

        request = SpeechRequest(text=text, language_code= language_code)

        lang = request.language_code

        tokenizer, model = self._get_mms_model(lang)

        try:
            logging.info(f"Generating speech for language '{lang}'")

            normalized_text = self.normalize_speech_text(request.text.lower())

            # break text into conversational chunks
            chunked_text = self.conversational_chunks(normalized_text)

            # add pauses for speech rhythm
            processed_text = self.add_speech_pauses(chunked_text)
            
            sentences = re.split(r'[.!?]+', processed_text)

            audio_chunks = []

            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue

                inputs = tokenizer(sentence, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    output = model(**inputs)

                waveform = output.waveform.squeeze().cpu().numpy()
                audio_chunks.append(waveform)

                # small pause b/w sentences.
                pause = np.zeros(int(0.08 * model.config.sampling_rate))
                audio_chunks.append(pause)

            audio = self.crossfade_audio(audio_chunks)

            # Squeeze() -> Removes batch dimension
            # cpu() -> Ensures tensor on CPU
            # numpy() -> Convert to numpy, Results in an array of audio samples
            
            peak = np.max(np.abs(audio))
            if peak > 0:
                audio = audio / peak

            # Prevent clipping
            audio = np.clip(audio, -0.99, 0.99)

            # Silence padding to prevent clipping
            pad_duration = int(0.15 * model.config.sampling_rate)   # 150 ms
            silence = np.zeros(pad_duration, dtype=np.float32)
            audio = np.concatenate([silence, audio, silence])

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
