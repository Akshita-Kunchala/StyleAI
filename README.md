# StyleSense – AI Fashion Recommendation System

## Features
- Skin tone detection using OpenCV
- AI outfit generation using Gemini
- Image generation using Stable Diffusion
- Personalized recommendations based on:
  - Occasion
  - Location
  - Budget
  - Style preferences
- Trend-aware suggestions
- Virtual try-on (AI generated)
- Shopping platform integration

## Setup

1. Clone repo
git clone <repo_link>

2. Install dependencies
pip install -r requirements.txt

3. Add API key
Create .env file:
GEMINI_API_KEY=your_key

4. Login HuggingFace
huggingface-cli login

5. Run
streamlit run app.py