# Voicedesk

**Voicedesk** is a multilingual, voice-powered assistant designed to handle phone calls, book appointments, and interact with users naturally. It supports **English, Gujarati, and Hindi**, enabling a seamless conversational experience natively across various languages.

The project processes audio files, transcribes speech to text, understands intents (like booking an appointment), extracts relevant entities (e.g., Name, Phone, Date, Time), generates natural language responses, and converts them back to speech using local TTS.

## Features

* **Multilingual Speech Recognition (ASR):** Capable of auto-detecting and transcribing English, Gujarati, and Hindi.
* **Intelligent Conversation Pipeline:** Leverages LangChain and Groq LLMs for fast intent classification, entity extraction, and conversational dialogue management.
* **Booking System:** Organizes and validates extracted information (Name, Phone number, Date, and Time).
* **Text-to-Speech (TTS):** Generates high-quality spoken responses to communicate back to the user seamlessly.
* **Dual Interfaces:**
  * **Streamlit App (`app.py`):** An intuitive web frontend for testing voice inputs, viewing extracted cards, and playing back the generated response.
  * **FastAPI Backend (`backend.py`):** A robust REST API (`/process-voice`) for integration with external applications (e.g., React frontend or telephony systems).
* **Database Integration:** Readily configurable with MongoDB to persist booking logs and conversational states.

## Tech Stack

* **Frameworks:** Streamlit, FastAPI
* **AI/ML:** Faster-Whisper, LangChain, Transformers, Torch, ONNX Runtime
* **LLM Provider:** Groq
* **Audio Processing:** Pydub, Soundfile, Torchaudio
* **Database:** MongoDB (PyMongo)

## Project Structure

```bash
voicedesk/
├── app.py                      # Streamlit frontend application
├── backend.py                  # FastAPI backend server
├── config.py                   # Configuration and environment variables
├── requirements.txt            # Python dependencies
├── agent/
│   └── voice_agent.py          # Orchestrates the ASR -> NLP -> TTS pipeline
├── services/                   # Core microservices logic
│   ├── asr_service.py          # Automatic Speech Recognition logic
│   ├── intent_service.py       # Intent classification using LLM
│   ├── entity_extractor.py     # Extracts structured data (Name, Date, etc.)
│   ├── dialogue_manager.py     # Tracks conversation state and missing entities
│   ├── response_generator.py   # Formats LLM responses into natural dialogue
│   ├── booking_service.py      # Validates and saves bookings
│   └── tts_service.py          # Text-to-Speech synthesis
└── models/
    └── schemas.py              # Pydantic models (e.g., FastApi Request/Response)
```

## Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd voicedesk
```

### 2. Create a Virtual Environment and Install Dependencies
Requires **Python 3.9+**.

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

*(Note: Depending on your system and requirements, you may need to install additional system libraries for audio processing like `ffmpeg`).*

### 3. Environment Variables `.env`
Create a `.env` file in the root directory mapping necessary secrets, such as:

```env
GROQ_API_KEY=your_groq_api_key_here
MONGO_URI=your_mongodb_connection_string
```

## Running the Application

### Option A: Run the Streamlit Interface (Testing & Demo)
This launches a web UI to upload audio files and visualize the agent's real-time extraction matrix.

```bash
streamlit run app.py
```

### Option B: Run the FastAPI Backend (Production / API Mode)
This runs the REST API to handle incoming audio files and respond programmatically.

```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```
- Interactive API Docs: `http://localhost:8000/docs`

## License

This project is licensed under the [MIT License](LICENSE).
