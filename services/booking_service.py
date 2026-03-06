import logging

from datetime import datetime
from typing import Dict, Any
import streamlit as st

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import certifi
from config import get_secret

@st.cache_resource
def get_mongo_client():

    mongo_uri= get_secret("MONGO_URI")

    if not mongo_uri:
        raise ValueError("MONGO_URI is not configured")

    client= MongoClient(
        mongo_uri,
        tls=True,
        tlsCAFile=certifi.where(),
        retryWrites=True,
        retryReads=True,
        maxPoolSize=10,
        minPoolSize=1,
        serverSelectionTimeoutMS=5000,
        connectTimeoutMS=20000,
        socketTimeoutMS=20000,
    )

    client.admin.command("ping")

    return client


class BookingService:
    def __init__(self):
        logging.info("Initializing BookingService (Simulation)")
        # Simulating a simple in-memory database of bookings

        self.use_memory= False
        self.memory_bookings= []


        db_name = get_secret("DB_NAME","ai-receptionist")

        try:
            self.client= get_mongo_client()

            self.db= self.client[db_name]

            self.collection = self.db["appointments"]

        # Ensure unique index to prevent double booking
            try:
                self.collection.create_index(
                    [("date",1),("time",1)],
                    unique=True,
                    background=True
                )
            except Exception as e:
                logging.warning(f"Index creation skipped: {e}")
        except Exception as e:
            logging.warning(f"MongoDB unavailable. Using in-memory fallback. Error: {e}")

            self.use_memory = True

    def check_availability(self, date: str, time: str) -> bool:
        """
        Checks if a specific date and time slot is available.
        In this simulation, we'll just check against our in-memory list.
        """
        logging.debug(f"Checking availability for {date} at {time}")
        
        if self.use_memory:

            for booking in self.memory_bookings:
                if booking["date"] == date and booking["time"] == time:
                    return False
            return True
        booking= self.collection.find_one({
            "date": date,
            "time":time,
        })

        return booking is None

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
        
        if not self.check_availability(date,time):
            return{
                "success":False,
                "message": "The requested time slot is unavailable."
            }

        booking_doc= {
            "name":name,
            "phone":phone,
            "date":date,
            "time":time,
            "status": "CONFIRMED",
            "created_at": datetime.utcnow(),
        }
        
        try:
            if self.use_memory:
                booking_doc["id"] = len(self.memory_bookings) + 1
                self.memory_bookings.append(booking_doc)
           
            else:
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