import logging
import os
from datetime import datetime
from typing import Dict, Any

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import certifi

from dotenv import load_dotenv

class BookingService:
    def __init__(self):
        logging.info("Initializing BookingService (Simulation)")
        # Simulating a simple in-memory database of bookings
        load_dotenv()

        mongo_uri= os.getenv("MONGO_URI")
        db_name= os.getenv("DB_NAME","ai_receptionist")

        if not mongo_uri:
            raise ValueError("MONGO_URI is not found ")

        self.client = MongoClient(
            mongo_uri,
            tls=True,
            tlsCAFile=certifi.where(),
            )
        self.db= self.client[db_name]

        self.collection = self.db["appointments"]

        # Ensure unique index to prevent double booking

        self.collection.create_index(
            [("date",1),("time",1)],
            unique=True,
        )

    def check_availability(self, date: str, time: str) -> bool:
        """
        Checks if a specific date and time slot is available.
        In this simulation, we'll just check against our in-memory list.
        """
        logging.debug(f"Checking availability for {date} at {time}")
        
        booking = self.collection.find_one({
            "date": date,
            "time":time,
        })

    def book_appointment(self, details: Dict[str, Any]) -> dict:
        """
        Creates a booking if the slot is available.
        """
        date = details.get("date")
        time = details.get("time")
        name = details.get("name")
        phone = details.get("phone")

        if not all([date, time, name, phone]):
            logging.error("Missing details for booking")
            return {"success": False, "message": "Missing necessary booking details."}
        
        booking_doc= {
            "name":name,
            "phone":phone,
            "date":date,
            "time":time,
            "status": "CONFIRMED",
            "created-at": datetime.utcnow(),
        }
        
        try:
            result = self.collection.insert_one(booking_doc)

            booking_doc["id"] = str(result.inserted_id)

            logging.info(f"Booking confirmed: {booking_doc}")

            return {
                "success": True,
                "message": "Booking confirmed",
                "booking": booking_doc,
            }
        except DuplicateKeyError:
            logging.warning(f"Slot already booked : {date} {time}")

            return {
                "success": False,
                "message": "The requested time slot is unavailable."
            }
        except Exception as e:
            logging.error(f"Booking failed: {e}")

            return {
                "success": False,
                "message": "Booking failed due to server error."
            }