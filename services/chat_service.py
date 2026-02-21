import os
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

MODEL = "models/gemini-2.5-flash"

SYSTEM_PROMPT = """
You are StyleAI, a friendly and professional personal fashion assistant.

Give short, helpful fashion advice.
Be practical and modern.
Keep responses under 4 sentences.
"""


def _call_gemini(prompt, retries=3):
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            error_text = str(e)

            if "429" in error_text:
                time.sleep(8)
            else:
                raise e

    return "⚠️ StyleAI is busy. Please wait a few seconds and try again."


def chat_response(message: str) -> str:
    prompt = f"""
{SYSTEM_PROMPT}

User: {message}
StyleAI:
"""
    return _call_gemini(prompt)