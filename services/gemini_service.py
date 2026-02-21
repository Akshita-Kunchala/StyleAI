import os
import json
import time
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Use ONE stable model only
MODEL = "models/gemini-2.5-flash"


def _call_gemini(prompt, retries=3):
    """
    Safe Gemini call with retry for 429 errors
    """
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )
            return response.text

        except Exception as e:
            error_text = str(e)

            # Rate limit handling
            if "429" in error_text:
                time.sleep(10)
            else:
                raise e

    raise Exception("Gemini rate limit exceeded. Please try again.")


def get_style_recommendation(data):
    prompt = f"""
You are StyleAI, a professional fashion stylist.

User details:
Age: {data.get('age')}
Gender: {data.get('gender')}
Skin tone: {data.get('skin_tone')}
Body type: {data.get('body_type')}
Hair type: {data.get('hair')}
Occasion: {data.get('occasion')}
Style preference: {data.get('style')}
Priority: {data.get('priority')}
Budget: {data.get('budget_min')} to {data.get('budget_max')}
Country: {data.get('country')}
State: {data.get('state')}
Color preference: {data.get('colors')}

Give highly personalized fashion recommendations.

Return ONLY JSON in this format:
{{
"outfit": "",
"makeup": "",
"hairstyle": "",
"why": "",
"trend": "",
"image_prompts": ["", "", ""]
}}
"""

    text = _call_gemini(prompt)

    # Extract JSON safely
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1 or end == -1:
        raise Exception("Invalid response from AI")

    return json.loads(text[start:end])