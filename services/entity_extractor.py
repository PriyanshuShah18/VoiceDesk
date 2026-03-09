import os
import logging
import dateparser
import re

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field, field_validator
from typing import Optional

class ExtractedEntities(BaseModel):
    name: Optional[str] = Field(None, description="The name of the client if mentioned, otherwise null")
    phone: Optional[str] = Field(None, description="The contact phone number of the client if mentioned, otherwise null")
    date_time_mention: Optional[str] = Field(None, description="The raw text phrase mentioning the requested date and/or time, otherwise null")

# Normalize Contact numbers

    @field_validator("phone", mode="before")
    def normalize_contact(cls,value):
        if not value:
            return None

        digits = "".join(c for c in str(value) if c.isdigit())

        if digits.startswith("91") and len(digits) ==12:
            digits= digits[2:]
        
        if len(digits) ==10:
            return digits
        return None

    # Sanity check for names
    @field_validator("name")
    def validate_name(cls,value):
        if not value:
            return None
        words = value.split()

        # Avoid hallucinated long names
        if len(words) >3:
            return None
        return value

class EntityExtractor:
    def __init__(self, model_name=None):
        """
        Initializes the entity extraction service.
        Checks for GROQ_API_KEY for cloud mode, otherwise local Ollama.
        """
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if groq_api_key:
            model = model_name or "llama-3.3-70b-versatile"
            logging.info(f"Initializing EntityExtractor with Groq model: {model}")
            self.llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model)
        else:
            model = model_name or "llama3"
            logging.info(f"Initializing EntityExtractor with Local Ollama model: {model}")
            # Use JSON format for reliable parsing
            self.llm = ChatOllama(model=model, format="json", temperature=0.0)
        self.parser = JsonOutputParser(pydantic_object=ExtractedEntities)
        
        prompt_template = """
You are an AI assistant that extracts specific information from user text for an appointment booking system.

The user may speak in Gujarati, Hindi or English.

Extract:
- client's name,
- phone number,
- any phrase mentioning date or time.

If a field is not mentioned return null.

Return the result strictly as a valid JSON object matching this schema:
{format_instructions}

User language: {language}
User input: "{text}"
"""
        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text","language"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        self.chain = self.prompt | self.llm | self.parser  # LCEL (Langchain Expression Language).

    def _extract_phone_regex(self, text:str):
        """
        Fallback phone extraction using regex when LLM misses the number.
        """
        pattern = r"(?:\+91[\-\s]?)?[6-9]\d{9}"
        match = re.search(pattern,text)

        if match:       
            digits = "".join(c for c in match.group() if c.isdigit())

            # Remove Indian country code
            if digits.startswith("91") and len(digits) ==12:
                digits = digits[2:]
            
            if len(digits) ==10:
                return digits
        return None


    def extract_entities(self, text: str, language: str = "en",previous_state: Optional[dict]=None) -> dict:
        """
        Extracts name, phone, date, and time from natural language text.
        """
        logging.debug(f"Extracting entities | text={text} | language={language}")
        try:
            # Get the structured JSON from LLM (it is already parsed into a dict by JsonOutputParser based on Pydantic schema)
            parsed_dict = self.chain.invoke({"text": text, "language": language})
            logging.debug(f"LLM Extraction Result: {parsed_dict}")
            
            # Use Pydantic to ensure the dictionary strictly matches what we expect
            validated_entities = ExtractedEntities(**parsed_dict)

            regex_phone = self._extract_phone_regex(text)
            
            final_entities = {
                "name": validated_entities.name,
                "phone": regex_phone or validated_entities.phone,
                "date": None,
                "time": None
            }
            # Parse the datetime string using dateparser
            dt_phrase = validated_entities.date_time_mention
            parsed_dt= None

            if dt_phrase:

                lang_map= {
                    "gu": ["gu"],
                    "hi": ["hi"],
                    "en": ["en"],
                }

                languages= lang_map.get(language, ["en"])

                clean_phrase=(
                    dt_phrase.lower()
                    .replace("around", "")
                    .replace("maybe", "")
                )

                parsed_dt = dateparser.parse(
                    clean_phrase,
                    languages=languages,
                    settings={
                        "PREFER_DATES_FROM": "future",
                        "RETURN_AS_TIMEZONE_AWARE": False,
                        "DATE_ORDER" : "DMY"
                    }
                )
            if parsed_dt:
                final_entities["date"] = parsed_dt.strftime("%Y-%m-%d")
                final_entities["time"] = parsed_dt.strftime("%H:%M")
            
            # MULTI TURN ENTITY MERGING
            if previous_state:
                for key in ["name","phone","date","time"]:
                    if not final_entities.get(key) and previous_state.get(key):
                        final_entities[key] = previous_state[key]

            return final_entities
            
        except Exception as e:
            logging.error(f"Error during entity extraction: {e}")
            return {
                "name": None,
                "phone": None,
                "date": None,
                "time": None
            }
