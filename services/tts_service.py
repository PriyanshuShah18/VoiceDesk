import logging
from transformers import VitsModel, AutoTokenizer
import torch
import os
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')
    
    text: str = Field(description="The text to synthesize")
    language_code: str = Field(default="gu", description="The ISO code of desired output language")
    
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


class TTSService:
    def __init__(self, output_dir="output_audio"):
        """
        Initializes the TTS service.
        """
        logging.info("Initializing TTS Service with local Meta MMS fallback")
        self.output_dir = output_dir
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

    def _get_mms_model(self, lang_code):
        """
        Lazily loads the requested facebook model.
        """
        if lang_code not in self.model_map:
            logging.warning(f"No local TTS model for language '{lang_code}', falling back to 'en'")
            lang_code = "en"
            
        if lang_code not in self.loaded_models:
            model_id = self.model_map[lang_code]
            logging.info(f"Loading local TTS model: {model_id}")
            try:
                self.loaded_tokenizers[lang_code] = AutoTokenizer.from_pretrained(model_id)
                self.loaded_models[lang_code] = VitsModel.from_pretrained(model_id)
            except Exception as e:
                logging.error(f"Failed to load local model {model_id}: {e}")
                return None, None
                
        return self.loaded_tokenizers[lang_code], self.loaded_models[lang_code]

    def generate_speech(self, text: str, output_filename="response.wav", language_code="gu") -> str:
        """
        Generates speech from text and saves it to a WAV file using local Meta MMS.
        """
        return self._generate_speech_mms(text, output_filename, language_code)

    def _generate_speech_mms(self, text: str, output_filename="response.wav", language_code="gu") -> str:
        # Validate through Pydantic model
        request = SpeechRequest(text=text, language_code=language_code)
        
        effective_lang = request.language_code
        logging.info(f"Generating speech using local MMS model for '{effective_lang}'...")
            
        tokenizer, model = self._get_mms_model(effective_lang)
        if not model:
            logging.error("TTS model not available. Cannot generate speech.")
            return None
            
        try:
            inputs = tokenizer(text, return_tensors="pt")
            
            with torch.no_grad():
                output = model(**inputs).waveform
                
            # Save audio
            output_path = os.path.join(self.output_dir, output_filename)
            sample_rate = model.config.sampling_rate
            
            # Save using scipy for better compatibility
            import scipy.io.wavfile as wavfile
            # Convert to numpy and handle float32 format
            audio_data = output.cpu().numpy()
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze()
            
            wavfile.write(output_path, sample_rate, audio_data)
            
            logging.info(f"Speech generated and saved to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error during speech generation: {e}")
            return None
