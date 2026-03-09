import os
import logging
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, field_validator

SUPPORTED_INTENTS = [
    "BOOK_APPOINTMENT",
    "CHECK_SLOTS",
    "RESCHEDULE",
    "CANCEL",
    "UNKNOWN"
]

class IntentResult(BaseModel):
    intent: str = Field(description="One of : BOOK_APPOINTMENT, CHECK_SLOTS, RESCHEDULE, CANCEL, UNKNOWN")
    
    @field_validator("intent", mode="before")
    def validate_and_sanitize_intent(cls, v):
        if not v:
            return "UNKNOWN"
            
        # Clean up output string to pure uppercase alphameric
        intent = str(v).strip().upper()
        intent = ''.join(c for c in intent if c.isalpha() or c == '_')
        
        if intent not in SUPPORTED_INTENTS:
            logging.warning(f"Unexpected intent generated: {intent}")
            return "UNKNOWN"
            
        return intent

class IntentService:
    def __init__(self, model_name=None):
        """
        Initializes the intent classification service.
        Checks for GROQ_API_KEY to use cloud-based Groq, otherwise defaults to local Ollama.
        """
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if groq_api_key:
            model = model_name or "llama-3.3-70b-versatile"
            logging.info(f"Initializing IntentService with Groq model: {model}")
            self.llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name=model)
        else:
            model = model_name or "llama3"
            logging.info(f"Initializing IntentService with Local Ollama model: {model}")
            self.llm = ChatOllama(model=model, temperature=0.0)
        
        self.supported_intents = SUPPORTED_INTENTS
        
        prompt_template = """
You are an AI receptionist for a service business.

Your task is to classify the user's intent based on their transcribed text.

The available intents are:
- BOOK_APPOINTMENT: The user wants to book a new appointment or consultation.
- CHECK_SLOTS: The user is asking about availability or open slots.
- RESCHEDULE: The user wants to change the date or time of an existing appointment.
- CANCEL: The user wants to cancel an appointment.
- UNKNOWN: The user's input does not match any of the above intents or is unclear.

Return ONLY a JSON object following this schema:

{format_instructions}

User input:
{text}
"""
        self.parser= JsonOutputParser(pydantic_object=IntentResult)

        self.prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["text"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            }
        )
        self.chain = self.prompt | self.llm | self.parser

    def detect_intent(self, text: str) -> str:
        """
        Predicts the intent from the given text.
        """
        logging.debug(f"Detecting intent for text: '{text}'")
        try:
            result = self.chain.invoke({"text": text})
            
            # Delegate validation and sanitization to Pydantic
            return result["intent"]
            
        except Exception as e:
            logging.error(f"Error during intent detection: {e}")
            return "UNKNOWN"
