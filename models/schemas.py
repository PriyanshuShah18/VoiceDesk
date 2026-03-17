from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
import logging
from typing import Literal

# TTS 

class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra='ignore')

    text:  str= Field(description="Text to synthesize")
    language_code: str = Field(default="en")

    @model_validator(mode="after")
    def validate_script_safety(self):
        latin_chars = sum(1 for c in self.text if c.isalpha() and c.isascii())
        total_chars = sum(1 for c in self.text if not c.isspace()) or 1

        if latin_chars / total_chars > 0.5 and self.language_code in ["gu", "hi"]:
            logging.warning("Fallback to English TTS")
            self.language_code = "en"

        return self

# ASR
class ASRRequest(BaseModel):
    audio_path: str
    language: str | None = None


class ASRResponse(BaseModel):
    text: str
    language : Literal["en","hi","gu"]

    @field_validator("language", mode="before")
    def validate_language(cls, v, info):
        text = info.data.get("text", "")

        latin = sum(1 for c in text if 'a' <= c.lower() <= 'z')
        indic = sum(1 for c in text if '\u0900' <= c <= '\u097F' or '\u0A80' <= c <= '\u0AFF')

        if indic > 2:
            if any ('\u0A80' <= c <= '\u0AFF' for c in text):
                return "gu"
            if any('\u0900' <= c <= '\u097F' for c in text):
                return "hi"

        result = v or "en"

        if result not in {"en", "hi", "gu"}:
            return "en"
        
        return result

# INTENT
SUPPORTED_INTENTS = [
    "BOOK_APPOINTMENT",
    "CHECK_SLOTS",
    "RESCHEDULE",
    "CANCEL",
    "CONFIRM",
    "UNKNOWN"
]

class IntentResponse(BaseModel):
    intent: str = Field(description="Detected intent")

    @field_validator("intent", mode="before")
    def validate_and_sanitize_intent(cls, v):
        if not v:
            return "UNKNOWN"

        intent = str(v).strip().upper()
        intent = ''.join(c for c in intent if c.isalnum() or c == '_')

        if intent not in SUPPORTED_INTENTS:
            logging.warning(f"Unexpected intent generated: {intent}")
            return "UNKNOWN"

        return intent


# ENTITY
class EntityResponse(BaseModel):
    name : str | None = None
    phone : str | None = None
    date : str | None = None
    time: str | None = None
    date_time_mention: str | None = None

    @field_validator("phone", mode="before")
    def normalize_contact(cls, value):
        if not value:
            return None
        
        digits = "".join(c for c in str(value) if c.isdigit())

        if digits.startswith("91") and len(digits) == 12:
            digits = digits[2:]

        if len(digits) == 10:
            return digits

        return None

    @field_validator("name")
    def validate_name(cls, value):
        if not value:
            return None

        words = value.strip().split()

        if len(words) == 0 or len(words) > 3:
            return None

        if any(any(char.isdigit() for char in word) for word in words):
            return None

        return value


# BOOKING
class BookingRequest(BaseModel):
    name: str
    phone: str
    date: str
    time: str

    @field_validator("phone")
    def validate_phone(cls, v):
        if len(v) != 10 or not v.isdigit():
            raise ValueError("Invalid phone number")
        return v

# API
class VoiceResponse(BaseModel):
    transcription: str
    intent: str
    entities: EntityResponse
    response_text: str
    audio_url: str | None = None