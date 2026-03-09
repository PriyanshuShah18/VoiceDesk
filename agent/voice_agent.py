import logging
import os

from services.asr_service import ASRService
from services.intent_service import IntentService
from services.entity_extractor import EntityExtractor
from services.dialogue_manager import DialogueManager
from services.booking_service import BookingService
from services.response_generator import ResponseGenerator
from services.tts_service import TTSService

class VoiceAgent:
    def __init__(self):
        logging.info("Initializing Voice Agent Pipeline...")
        # Initialize all services
        self.asr = ASRService()
        self.intent_service = IntentService()
        self.entity_extractor = EntityExtractor()
        self.dialogue_manager = DialogueManager()
        self.booking_service = BookingService()
        self.response_generator = ResponseGenerator()
        self.tts = TTSService()
        
        logging.info("Voice Agent Pipeline initialized successfully.")

    def process_audio(self, audio_path: str, forced_language: str = None) -> dict:
        """
        Runs the full pipeline from audio input to audio output.
        Returns a dictionary with all intermediate results.
        """
        logging.info(f"Processing New Audio Input: {audio_path}")
        
        # 1 & 2. ASR and Language Detection
        # Passing forced_language to ASR for routing
        transcription_result = self.asr.transcribe(audio_path, forced_language=forced_language)
        
        text = transcription_result["text"]
        language = transcription_result["language"]
        
        logging.info(f"Transcription ({language}): {text}")
        
        if not text:
            logging.warning("No text transcribed.")

            fallback = "I couldn't hear anything. Please try again."

            return {
                "transcription": "",
                "language": language,
                "intent": "NONE",
                "entities": {},
                "response_text": fallback,
                "audio_path": None
            }

        # 3. Intent Detection
        intent = self.intent_service.detect_intent(text)

        logging.info(f"Detected Intent: {intent}")

        # 4. Entity Extraction
        entities = self.entity_extractor.extract_entities(text)

        logging.info(f"Extracted Entities: {entities}")

        # 5. Dialogue Management
        self.dialogue_manager.update_state(intent, entities)

        next_action = self.dialogue_manager.get_next_action()
        
        logging.info(f"Next Action: {next_action}")

        # 6. Booking Logic (if applicable)
        booking_result = None
        
        if next_action["action"] == "confirm_booking":
            booking_result = self.booking_service.book_appointment(self.dialogue_manager.state)
            
            next_action["booking_result"] = booking_result
            # Reset conversation state after booking attempt
        
        if booking_result and booking_result.get("success"):
            self.dialogue_manager.reset_state()
            
            #self.dialogue_manager.reset_state()
            logging.info(f"Booking Result: {booking_result}")

        # 7. Response Generation
       # response_text = self.response_generator.generate_response(
       #     action_details=next_action, 
       #     state=self.dialogue_manager.state, 
       #     language=language
       # )
        response_text = text

        logging.info(f"Echo Response Text: {response_text}")
        
        logging.info(f"Response Text: {response_text}")

        if language not in ["en", "hi", "gu"]:
            language= "en"

        try:
            output_audio_path = self.tts.generate_speech(
                response_text,
                language_code= language
            )
        except Exception as e:
            logging.error(f"TTS failed: {e}")
            output_audio_path = None

        logging.info(
            f" Pipeline Completed. Output Audio: {output_audio_path}"
        )
        
        return {
            "transcription": text,
            "language": language,
            "intent": intent,
            "entities": entities,
            "response_text": response_text,
            "audio_path": output_audio_path
        }
