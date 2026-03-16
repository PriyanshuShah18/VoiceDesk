from gtts import gTTS
import os

def text_to_audio(text, filename="output", format="mp3", language="en"):
    """
    Convert text to speech and save as mp3 or wav
    """

    # Ensure format is valid
    if format not in ["mp3", "wav"]:
        raise ValueError("Format must be 'mp3' or 'wav'")

    temp_mp3 = filename + ".mp3"

    # Generate speech
    tts = gTTS(text=text, lang=language)
    tts.save(temp_mp3)

    # Convert to wav if required
    if format == "wav":
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(filename + ".wav", format="wav")
        os.remove(temp_mp3)
        print(f"Audio saved as {filename}.wav")
    else:
        print(f"Audio saved as {filename}.mp3")


# Example usage
text ="""
Hello, I want to book an appointment for Tomorrow at 5 pm for Web Design.
My name is Alex.
And my contact number is 1234567890.
"""

text_to_audio(text, filename="speech3", format="mp3")