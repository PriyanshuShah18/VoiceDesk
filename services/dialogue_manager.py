import logging

class DialogueManager:
    def __init__(self):
        logging.info("Initializing DialogueManager")
        self.state = {
            "intent": None,
            "date": None,
            "time": None,
            "name": None,
            "phone": None
        }
    def get_state(self):
        """
        Returns the current conversation state.
        Used by EntityExtactor for multi-turn merging.
        """
        return self.state

    def update_state(self, intent: str, entities: dict):
        """
        Updates the conversation state with new intent and entities.
        """
        logging.debug(f"Updating dialogue state. Intent: {intent}, Entities: {entities}")
        
        # Keep current intent if new intent is UNKNOWN or None, unless it's the very start
        if intent and intent != "UNKNOWN" and intent != self.state["intent"]:
            self.reset_state()
            self.state["intent"] = intent
            
        if entities:
            if entities.get("name"): self.state["name"] = entities.get("name")
            if entities.get("phone"): self.state["phone"] = entities.get("phone")
            if entities.get("date"): self.state["date"] = entities.get("date")
            if entities.get("time"): self.state["time"] = entities.get("time")
            
        logging.info(f"Current State: {self.state}")

    def get_missing_fields(self):
        """
        Returns a list of missing booking fields.
        """
        missing = []

        if not self.state.get("name"):
            missing.append("name")
        if not self.state.get("phone"):
            missing.append("phone")
        if not self.state.get("date"):
            missing.append("date")
        if not self.state.get("time"):
            missing.append("time")

        return missing

    def get_next_action(self) -> dict:
        """
        Determines the next conversational step based on current state.
        """
        intent = self.state.get("intent")
        
        if not intent:
            return {
                "action": "ask_intent",
                "field": "intent",
                "missing": ["intent"],
                "reason": "Need to know what the caller wants."
            }
            
        if intent == "BOOK_APPOINTMENT":
            missing_fields = self.get_missing_fields()
            
            if missing_fields:
                next_field = missing_fields[0]

                return {
                    "action": "ask_detail",
                    "field": next_field,
                    "missing": missing_fields,
                    "reason": f"Need user's {next_field} to continue booking."
                }
            else:
                return {
                    "action": "confirm_booking",
                    "details": {
                        "name": self.state["name"],
                        "date": self.state["date"],
                        "time": self.state["time"],
                    },
                    "reason": "All appointment details are collected"
                }
                
        if intent in ["CHECK_SLOTS", "RESCHEDULE", "CANCEL"]:
            return {
                "action": "handle_other_intent",
                "intent": intent,
                "reason": f"Managing request for {intent}"
            }
            
        return {
            "action": "ask_clarification",
            "field": "intent",
            "intent": intent,
            "reason": "User request unclear."
        }

    def reset_state(self):
        """
        Resets the conversation state.
        """
        logging.info("Resetting dialogue state")
        self.state = {
            "intent": None,
            "date": None,
            "time": None,
            "name": None,
            "phone": None
        }
