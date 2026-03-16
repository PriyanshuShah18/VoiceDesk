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
        self.llm = ChatGroq(temperature=0.65, groq_api_key=groq_api_key, model_name=model)
        
        prompt_template = """
You are a friendly receptionist answering calls for a service business.

Speak naturally like a human receptionist.

Important rules:
- Never repeat the user's transcript.
- Keep responses CONCISE and natural for spoken sentence.
- Use SHORT conversational sentences.
- Prefer 2-3 short sentences instead of one long sentence.
- Ask follow-up questions when information is missing.
- Speak politely and clearly because the response will be spoken aloud.
- Be conversational.
- Acknowledge the user's request briefly before asking the next question.
- Include the Contact Number in the response if available.
Conversation History:
{history}

User's language: {language}

Current conversation state:
{state}

Suggested system action:
{action_details}

Speaking style:
{style_hint}

Generate a natural conversational response based on the action.
Use natural spoken phrasing suitable for a voice assistant.

Language rules:
- If language = gu → respond in Gujarati script.
- If language = hi → respond in Hindi script.
- If language = en → respond in English.

Return ONLY the response sentence.
Do not include explanations or formatting.

Response:"""
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, action_details: dict, state: dict, language: str = "en", history=None) -> str:
        """
        Generates a natural language response.
        """
        logging.debug(f"Generating response. Action: {action_details}, Language: {language}")
        try:
            history_text = ""

            if history:
                history_lines = []
                for msg in history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    history_lines.append(f"{role}: {content}")

                history_text = "\n".join(history_lines)

            action = action_details.get("action")

            style_hint = ""

            if action == "ask_intent":
                style_hint = "Use a friendly greeting tone as if answering a phone call."

            elif action == "ask_detail":
                style_hint = "Use a helpful tone asking politely for the missing detail."

            elif action == "confirm_booking":
                style_hint = "Use a confirming tone that sounds reassuring and professional"

            elif action == "handle_other_intent":
                style_hint = "Use a helpful customer support tone."

            elif action == "ask_clarification":
                style_hint = "Use a polite clarification tone asking the caller to repeat."

            else:
                style_hint = "Use a natural conversational receptionist tone."

            action_text = f"""
            Action: {action_details.get("action")}
            Field needed : {action_details.get("field")}
            Missing fields: {action_details.get("missing")}
            Reason: {action_details.get("reason")}
            Details: {action_details.get("details")}
            """

            result = self.chain.invoke({
                "action_details": action_text,
                "state": str(state),
                "language": language,
                "history": history_text,
                "style_hint": style_hint
            })
            response_text = result.strip().replace("\n"," ")
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
