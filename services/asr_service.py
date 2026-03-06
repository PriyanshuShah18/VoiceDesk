import os
import logging
from faster_whisper import WhisperModel
import requests
from pydantic import BaseModel, Field, field_validator


class TranscriptionResponse(BaseModel):
    text: str = Field(description="The transcribed text")
    language: str = Field(description="The ISO 639-1 language code (e.g., 'en', 'gu', 'hi')")

    @field_validator("language", mode="before")
    def validate_language(cls, v, info):
        # We can extract text from the current model values being validated
        # 'data' in pydantic v2 is accessed via info.data
        if 'text' in info.data:
            text = info.data['text']
            
            latin_count = sum(1 for c in text if 'a' <= c.lower() <= 'z')
            investigate_indic = sum(1 for c in text if '\u0A80' <= c <= '\u0AFF' or '\u0900' <= c <= '\u097F')
            
            if investigate_indic > 2 and investigate_indic > (latin_count / 2):
                return "gu"
            else:
                return "en"
        return v or "en"


class ASRService:
    def __init__(self, model_size="medium", device="cpu", compute_type="int8"):
        """
        Initializes the ASR service.
        Checks for GROQ_API_KEY to use Groq's high-performance Cloud ASR.
        Otherwise falls back to local faster-whisper.
        """
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.hf_token = os.getenv("HF_TOKEN")
        self.hf_api_url = "https://api-inference.huggingface.co/models/ai4bharat/indic-conformer-600m-multilingual"
        
        if self.hf_token:
            logging.info("Initializing Hugging Face ASR Support (AI4Bharat IndicConformer)...")
        
        if self.groq_api_key:
            logging.info("Initializing Groq Cloud ASR (Whisper-v3)...")
            # We don't need to load a local model if Groq is available
            self.model = None
        else:
            logging.info(f"Loading local faster-whisper model '{model_size}' on {device}...")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            logging.info("Local ASR Model loaded successfully.")

    def transcribe(self, audio_path: str, forced_language: str = None, provider: str = "auto") -> dict:
        """
        Transcribes the given audio file using the specified provider.
        Providers: 'auto' (smart hybrid), 'groq', 'huggingface', 'local'
        """
        if provider == "smart" or provider == "auto":
            return self._transcribe_smart(audio_path, forced_language)
        elif provider == "huggingface" and self.hf_token:
            return self._transcribe_huggingface(audio_path, forced_language)
        elif (provider == "groq") and self.groq_api_key:
            return self._transcribe_groq(audio_path, forced_language)
        elif provider == "local" or self.model:
            return self._transcribe_local(audio_path, forced_language)
        else:
            # Fallback chain
            if self.hf_token: return self._transcribe_huggingface(audio_path, forced_language)
            return TranscriptionResponse(text="", language="en").model_dump()

    def _transcribe_smart(self, audio_path: str, forced_language: str = None) -> dict:
        """
        Smart Hybrid Mode: Uses Groq for initial detection. 
        If it's English, trust Groq. If it's Gujarati/Hindi, use HF for better accuracy.
        """
        logging.info("Running Smart Hybrid ASR...")
        
        # 1. Groq Pass for Detection and Fallback
        groq_result = self._transcribe_groq(audio_path, forced_language=forced_language)
        detected_lang = groq_result.get("language", "en")
        transcribed_text = groq_result.get("text", "")
        
        # 2. Routing Decision
        # If it's English, Groq is the best. Trust it.
        if detected_lang == "en":
            logging.info("Smart Hybrid: Detected English. Sticking with Groq.")
            return groq_result
            
        # If it's Gujarati and we have HF, get the high-accuracy version
        if detected_lang == "gu" and self.hf_token:
            logging.info("Smart Hybrid: Detected Gujarati. Routing to Hugging Face for high accuracy.")
            hf_result = self._transcribe_huggingface(audio_path, forced_language="gu")
            
            # Script Safety Check: If HF returns something that looks like gibberish 
            # compared to Groq's confidence, or if it's too short, return Groq's.
            if len(hf_result.get("text", "")) < 2 and len(transcribed_text) > 5:
                logging.warning("HF result too short, falling back to Groq.")
                return groq_result
                
            return hf_result
            
        return groq_result

    def _transcribe_groq(self, audio_path: str, forced_language: str = None) -> dict:
        logging.info(f"Transcribing via Groq Cloud ASR: {audio_path} (Forced: {forced_language})")
        try:
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {self.groq_api_key}"}
            
            with open(audio_path, "rb") as file:
                files = {
                    "file": (os.path.basename(audio_path), file),
                    "model": (None, "whisper-large-v3"),
                    "response_format": (None, "json"),
                    "prompt": (None, "The conversation is in Gujarati, Hindi or English.")
                }
                # Only add language parameter if explicitly forced, otherwise let Whisper auto-detect
                if forced_language:
                    files["language"] = (None, forced_language)
                response = requests.post(url, headers=headers, files=files)
            
            if response.status_code == 200:
                result = response.json()
                text = result.get("text", "").strip()
                
                # If we forced a language, return it
                if forced_language:
                    return TranscriptionResponse(text=text, language=forced_language).model_dump()
                
                # Let Pydantic validator handle the script detection
                response_obj = TranscriptionResponse(text=text, language="")
                logging.info(f"Groq Cloud Result - Pydantic detected language: {response_obj.language}")
                return response_obj.model_dump()
            else:
                logging.error(f"Groq ASR failed ({response.status_code}): {response.text}")
                return TranscriptionResponse(text="", language="en").model_dump()
        except Exception as e:
            logging.error(f"Error during Groq transcription: {e}")
            return TranscriptionResponse(text="", language="en").model_dump()

    def _transcribe_huggingface(self, audio_path: str, forced_language: str = None) -> dict:
        logging.info(f"Transcribing via Hugging Face ASR: {audio_path}")
        lang = forced_language if forced_language else "gu"
        
        try:
            with open(audio_path, "rb") as f:
                data = f.read()
            
            headers = {"Authorization": f"Bearer {self.hf_token}"}
            response = requests.post(self.hf_api_url, headers=headers, data=data)
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle both dict and list responses
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get("text", "").strip()
                elif isinstance(result, dict):
                    text = result.get("text", "").strip()
                else:
                    text = ""
                
                return TranscriptionResponse(text=text, language=lang).model_dump()
            elif response.status_code == 503:
                logging.warning("Hugging Face model is loading. Please wait a moment.")
                return TranscriptionResponse(text="Model is loading on Hugging Face. Please try again in 30 seconds.", language="en").model_dump()
            else:
                logging.error(f"Hugging Face ASR failed ({response.status_code}): {response.text}")
                return TranscriptionResponse(text="", language="en").model_dump()
        except Exception as e:
            logging.error(f"Error during Hugging Face transcription: {e}")
            return TranscriptionResponse(text="", language="en").model_dump()

    def _transcribe_local(self, audio_path: str, forced_language: str = None) -> dict:
        logging.debug(f"Transcribing locally from: {audio_path} (Forced: {forced_language})")
        
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=5, 
            language=forced_language,
            initial_prompt="Gujarati, Hindi, English conversation"
        )
        
        text = " ".join([segment.text.strip() for segment in segments])
        
        if forced_language:
            detected_lang = forced_language
        else:
            supported = ["gu", "hi", "en"]
            detected_lang = info.language
            if detected_lang not in supported:
                detected_lang = "en"
        
        return TranscriptionResponse(text=text.strip(), language=detected_lang).model_dump()
