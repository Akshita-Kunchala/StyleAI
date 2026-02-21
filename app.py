import streamlit as st
import numpy as np
from PIL import Image
import random
import pycountry

from services.skin_service import detect_skin_tone
from services.gemini_service import get_style_recommendation
from services.image_service import (
    generate_outfit_images,
    generate_pinterest_inspo,
    virtual_tryon,
)
from services.location_service import get_states
from services.chat_service import chat_response

# ══════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="StyleAI", layout="wide")

# ══════════════════════════════════════════════════════
# Styles
# ══════════════════════════════════════════════════════
st.markdown("""
<style>
.stApp {
    background: linear-gradient(-45deg, #0f172a, #1e293b, #111827, #020617);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
    color: white;
}
@keyframes gradientBG {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.main .block-container { background: transparent; }
.card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(14px);
    border-radius: 16px;
    padding: 25px;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
}
.badge-real  { background:#1e40af; color:#bfdbfe; padding:2px 8px; border-radius:20px; font-size:11px; }
.badge-ai    { background:#4c1d95; color:#ddd6fe; padding:2px 8px; border-radius:20px; font-size:11px; }
.badge-tryon { background:#14532d; color:#bbf7d0; padding:2px 8px; border-radius:20px; font-size:11px; }
.img-card {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════
def get_state(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]

# ══════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════
st.title("✨ StyleAI")
quotes = [
    "Style is a way to say who you are without speaking.",
    "Confidence is your best outfit.",
    "Elegance never goes out of style.",
    "Fashion fades, style is eternal.",
]
st.info(random.choice(quotes))

# ══════════════════════════════════════════════════════
# Step control
# ══════════════════════════════════════════════════════
if "step" not in st.session_state:
    st.session_state.step = 1

st.progress(st.session_state.step / 3)
countries = [c.name for c in pycountry.countries]


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — About You
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.step == 1:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Tell us about yourself")

    uploaded = st.file_uploader(
        "Upload your photo (used for virtual try-on)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, width=250, caption="Your photo")
        st.session_state.user_photo_bytes = uploaded.getvalue()
        st.session_state.skin_tone = detect_skin_tone(np.array(image))
        st.success(f"Skin tone detected: **{st.session_state.skin_tone}**")
    else:
        st.caption("No photo? No problem — fill in the details below.")
        st.session_state.user_photo_bytes = None

        st.markdown("""
        <div style="background:rgba(99,102,241,0.15);border-radius:8px;padding:8px 12px;
        margin-bottom:10px;font-size:12px;color:#a5b4fc;">
        🎤 <b>Voice input:</b> Click the text box below, then press
        <kbd>Win + H</kbd> (Windows) to speak your description.
        </div>
        """, unsafe_allow_html=True)

        st.session_state.description = st.text_area(
            "Describe yourself (optional)",
            value=get_state("description", ""),
            height=100,
        )
        st.session_state.skin_tone = st.selectbox(
            "Skin tone",
            ["Very Fair", "Fair", "Olive", "Dusky", "Deep"],
            index=["Very Fair", "Fair", "Olive", "Dusky", "Deep"].index(
                get_state("skin_tone", "Medium")
            ),
        )
        st.session_state.body_type = st.selectbox(
            "Body type",
            ["Slim", "Average", "Curvy", "Plus"],
            index=["Slim", "Average", "Curvy", "Plus"].index(
                get_state("body_type", "Average")
            ),
        )
        st.session_state.hair = st.selectbox(
            "Hair type",
            ["Straight", "Wavy", "Curly"],
            index=["Straight", "Wavy", "Curly"].index(get_state("hair", "Wavy")),
        )

    st.session_state.age = st.number_input("Age", 10, 80, value=get_state("age", 25))
    st.session_state.gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Other"],
        index=["Female", "Male", "Other"].index(get_state("gender", "Female")),
    )

    if st.button("Next →"):
        st.session_state.step = 2
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Preferences
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 2:

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("Your Preferences")

    st.session_state.occasion = st.selectbox(
        "Occasion",
        ["Casual", "Office", "Wedding", "Party", "Festival", "Date"],
        index=["Casual", "Office", "Wedding", "Party", "Festival", "Date"].index(
            get_state("occasion", "Casual")
        ),
    )
    st.session_state.style = st.selectbox(
        "Style",
        ["Minimalist", "Heavy", "Ethnic", "Streetwear", "Formal", "Trendy"],
        index=["Minimalist", "Heavy", "Ethnic", "Streetwear", "Formal", "Trendy"].index(
            get_state("style", "Minimalist")
        ),
    )
    st.session_state.priority = st.radio(
        "Priority",
        ["Comfort", "Fashion", "Both"],
        index=["Comfort", "Fashion", "Both"].index(get_state("priority", "Comfort")),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.session_state.budget_min = st.number_input(
            "Min Budget", value=get_state("budget_min", 0), step=500
        )
    with col2:
        st.session_state.budget_max = st.number_input(
            "Max Budget", value=get_state("budget_max", 20000), step=500
        )

    st.session_state.colors = st.multiselect(
        "Color preference",
        ["Red", "Blue", "Black", "White", "Pastel", "Earth tones"],
        default=get_state("colors", []),
    )

    default_country = get_state("country", "India")
    st.session_state.country = st.selectbox(
        "Country",
        countries,
        index=countries.index(default_country) if default_country in countries else 0,
    )

    states = get_states(st.session_state.country)
    if states:
        default_state = get_state("state", states[0])
        st.session_state.state = st.selectbox(
            "State",
            states,
            index=states.index(default_state) if default_state in states else 0,
        )
    else:
        st.session_state.state = st.text_input(
            "State / Region", value=get_state("state", "")
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("✨ Generate Style"):
            st.session_state.step = 3
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Style Report + Images
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.step == 3:

    # ── Style Recommendation ─────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("✨ Your Personalized Style Report")

    with st.spinner("Crafting your style recommendation…"):
        try:
            result = get_style_recommendation(st.session_state)
        except Exception as e:
            st.error(f"StyleAI is busy right now. Please go back and try again. ({e})")
            if st.button("← Back"):
                st.session_state.step = 2
                st.rerun()
            st.stop()

    st.subheader("👗 Outfit Recommendation")
    st.write(result["outfit"])

    st.subheader("💄 Makeup & 💇 Hairstyle")
    st.write(result["makeup"])
    st.write(result["hairstyle"])

    st.subheader("💡 Why this suits you")
    st.info(result["why"])

    st.subheader("📈 Trend Insight")
    st.success(result["trend"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 1. OUTFIT IMAGES ─────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("👗 Visual Preview — Outfit Looks")
    st.caption("AI-generated outfit images based on your recommendation")

    outfit_prompts = result.get("image_prompts", [result["outfit"]])[:3]
    style_ctx = f"{st.session_state.get('style','')} {st.session_state.get('occasion','')}"

    with st.spinner("Generating outfit images… (this takes ~30–60s on first load)"):
        outfit_results = generate_outfit_images(outfit_prompts, style_context=style_ctx)

    if outfit_results:
        cols = st.columns(len(outfit_results))
        for i, item in enumerate(outfit_results):
            with cols[i]:
                if item["image"] is not None:
                    st.image(item["image"], use_container_width=True, caption=item["prompt"][:60])
                else:
                    st.markdown("""
                    <div style='background:rgba(239,68,68,0.1);border:1px solid #ef4444;
                    border-radius:10px;padding:20px;text-align:center;color:#fca5a5;'>
                    ⚠️ Could not generate.<br><small>Check HF_TOKEN & internet.</small>
                    </div>""", unsafe_allow_html=True)
    else:
        st.error("Image generation unavailable. Verify your HF_TOKEN in `.env` has Inference API access.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 2. PINTEREST INSPO BOARD ─────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Pinterest Inspo Board")
    st.caption("Real photos + AI-generated mood board for your style")

    trend_keyword  = result.get("trend", "fashion aesthetic")[:40]
    style_keyword  = f"{st.session_state.get('style','')} {st.session_state.get('occasion','')} outfit"
    color_keyword  = " ".join(st.session_state.get("colors", []))
    inspo_keywords = [k for k in [trend_keyword, style_keyword, color_keyword] if k.strip()][:2]

    with st.spinner("Building your inspo board…"):
        inspo_results = generate_pinterest_inspo(
            inspo_keywords,
            n_real=2,
            n_generated=2,
        )

    if inspo_results:
        chunk_size = 4
        for row_start in range(0, len(inspo_results), chunk_size):
            row = inspo_results[row_start : row_start + chunk_size]
            cols = st.columns(len(row))
            for j, item in enumerate(row):
                with cols[j]:
                    badge = (
                        '<span class="badge-real">📷 Real</span>'
                        if item["source"] == "unsplash"
                        else '<span class="badge-ai">✨ AI</span>'
                    )
                    st.markdown(badge, unsafe_allow_html=True)
                    if item["image"] is not None:
                        st.image(item["image"], use_container_width=True, caption=item["keyword"][:40])
                    else:
                        st.markdown("""
                        <div style='background:rgba(255,255,255,0.05);border-radius:10px;
                        padding:30px;text-align:center;color:#94a3b8;'>
                        No image
                        </div>""", unsafe_allow_html=True)
    else:
        st.warning("No inspo images generated. Add UNSPLASH_ACCESS_KEY to .env for real photos.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3. VIRTUAL TRY-ON ────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🪄 Virtual Try-On")

    user_photo_bytes = st.session_state.get("user_photo_bytes")

    if user_photo_bytes:
        from io import BytesIO
        user_photo = Image.open(BytesIO(user_photo_bytes)).convert("RGB")

        st.markdown('<span class="badge-tryon">✅ Using your uploaded photo</span>', unsafe_allow_html=True)
        st.write("")

        outfit_desc = result["outfit"]
        hair_makeup = f"{result.get('hairstyle', '')}. {result.get('makeup', '')}"
        accessories = ""

        with st.spinner("Creating your virtual try-on… (30–90s)"):
            tryon = virtual_tryon(
                user_photo=user_photo,
                outfit_description=outfit_desc,
                hair_makeup_description=hair_makeup,
                accessories=accessories,
                use_ai_compositing=True,
            )

        if tryon["success"]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📸 Your Photo**")
                st.image(user_photo, use_container_width=True)
            with col2:
                st.markdown("**✨ Virtual Try-On**")
                st.image(tryon["tryon_image"], use_container_width=True)

            method = tryon.get("method", "")
            if method == "ai_img2img":
                st.caption("✅ AI-powered compositing applied.")
            elif method == "blend_fallback":
                st.caption("💡 Preview blend used. For higher quality, ensure your HF token has full Inference API access.")
        else:
            st.error("Virtual try-on failed. Showing outfit recommendation only.")
            if outfit_results and outfit_results[0]["image"]:
                st.image(outfit_results[0]["image"], caption="Recommended outfit", width=350)

    else:
        st.markdown("""
        <div style='background:rgba(99,102,241,0.1);border:1px dashed #6366f1;
        border-radius:12px;padding:30px;text-align:center;'>
        <p style='font-size:18px;'>📸 Upload your photo in <strong>Step 1</strong> to see the virtual try-on!</p>
        <p style='color:#94a3b8;font-size:13px;'>We'll overlay the recommended outfit, hair & makeup onto your photo.</p>
        </div>
        """, unsafe_allow_html=True)

        if outfit_results and any(r["image"] for r in outfit_results):
            st.write("")
            st.caption("Here's how the outfit looks on a model instead:")
            best = next((r for r in outfit_results if r["image"]), None)
            if best:
                st.image(best["image"], width=350)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Shop Links ────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🛍️ Shop Similar Styles")
    query = result["outfit"].replace(" ", "+")[:80]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"[🛍️ Myntra](https://www.myntra.com/{query})")
        st.markdown(f"[📦 Amazon](https://www.amazon.in/s?k={query})")
    with col2:
        st.markdown(f"[✨ Ajio](https://www.ajio.com/search/?text={query})")
        st.markdown(f"[🛒 Flipkart](https://www.flipkart.com/search?q={query})")
    with col3:
        st.markdown(f"[💄 Nykaa Fashion](https://www.nykaafashion.com/catalogsearch/result/?q={query})")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("← Back"):
        st.session_state.step = 2
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# Floating Chat  ← ONLY THIS SECTION CHANGED
# ════════════════════════════════════════════════════════════════════════════
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button("💬 StyleAI Chat", key="chat_toggle"):
    st.session_state.chat_open = not st.session_state.chat_open

if st.session_state.chat_open:
    st.sidebar.header("💬 StyleAI Assistant")

    # Chat history display
    for chat in st.session_state.chat_history[-6:]:
        role_icon = "🧑" if chat["role"] == "user" else "✨"
        bg = "rgba(255,255,255,0.05)" if chat["role"] == "user" else "rgba(99,102,241,0.15)"
        st.sidebar.markdown(f"""
        <div style="background:{bg};border-radius:8px;padding:8px 10px;
        margin:4px 0;font-size:13px;">
        {role_icon} {chat["text"]}
        </div>""", unsafe_allow_html=True)

    # Input
    msg = st.sidebar.text_input(
        "Ask about fashion…",
        placeholder="e.g. What shoes go with a saree?",
        key="chat_input",
    )

    col1, col2 = st.sidebar.columns([2, 1])
    with col1:
        send = st.button("Send ➤", key="chat_send", use_container_width=True)
    with col2:
        if st.button("Clear", key="chat_clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if send and msg.strip():
        st.session_state.chat_history.append({"role": "user", "text": msg})
        with st.spinner("Thinking…"):
            reply = chat_response(msg)
        st.session_state.chat_history.append({"role": "assistant", "text": reply})
        st.rerun()