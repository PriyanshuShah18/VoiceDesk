import os
import logging
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import get_secret

class ResponseGenerator:
    def __init__(self, model_name=None):
        """
        Initializes the response generation service.
        Uses Groq Cloud LLM for natural language generation.
        """
        groq_api_key = get_secret("GROQ_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not configured in secrets or environment.")
            
        model = model_name or "llama-3.3-70b-versatile"
        logging.info(f"Initializing ResponseGenerator with Groq: {model}")
        self.llm = ChatGroq(temperature=0.4, groq_api_key=groq_api_key, model_name=model)
        
        prompt_template = """
You are a helpful receptionist for a service business.
Your goal is to converse naturally with the user based on the current context.
Keep your responses concise as they will be spoken aloud to the user over phone/audio.

User's language: {language}
Current conversation state: {state}
Suggested system action: {action_details}

Generate a friendly conversational response matching the 'Suggested system action' in the user's language (if 'gu', use Gujarati script; if 'hi', use Hindi script; if 'en', use English).
Do not include any translations or extra text, JUST the response.

Response:"""
        
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, action_details: dict, state: dict, language: str = "en") -> str:
        """
        Generates a natural language response.
        """
        logging.debug(f"Generating response. Action: {action_details}, Language: {language}")
        try:
            result = self.chain.invoke({
                "action_details": str(action_details),
                "state": str(state),
                "language": language
            })
            response_text = result.strip()
            logging.info(f"Generated response: {response_text}")
            return response_text
        except Exception as e:
            logging.error(f"Error during response generation: {e}")
            if language == "gu":
                return "માફ કરશો, મને સમજાયું નહીં."
            elif language == "hi":
                return "माफ़ कीजिए, मुझे समझ नहीं आया।"
            else:
                return "I'm sorry, I didn't catch that."
