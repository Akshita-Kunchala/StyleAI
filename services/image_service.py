import os
import time
import base64
import logging
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFilter
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN     = os.getenv("HF_TOKEN")
UNSPLASH_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
HF_HEADERS   = {"Authorization": f"Bearer {HF_TOKEN}"}
HF_BASE_URL  = "https://router.huggingface.co/hf-inference/models"

# Fast models — sd-turbo generates in ~3-5s vs 60s+ for SD 2.1
FAST_MODEL   = "black-forest-labs/FLUX.1-schnell"
INPAINT_MODEL = "stabilityai/stable-diffusion-2-inpainting"

# FLUX.1-schnell — fastest model WITH active Inference Provider on free tier
TURBO_PARAMS = {"num_inference_steps": 4, "guidance_scale": 3.5}

TIMEOUT     = 30
MAX_RETRIES = 2
RETRY_SLEEP = 8


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_keep_aspect(img: Image.Image, max_side: int = 512) -> Image.Image:
    w, h = img.size
    scale = min(max_side / w, max_side / h)
    if scale >= 1:
        return img
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def _make_placeholder(text: str, size=(400, 400)) -> Image.Image:
    img = Image.new("RGB", size, color=(40, 40, 50))
    draw = ImageDraw.Draw(img)
    draw.text((size[0] // 2, size[1] // 2), text, fill=(120, 120, 130), anchor="mm")
    return img


# ═════════════════════════════════════════════════════════════════════════════
# CORE HF CALL
# ═════════════════════════════════════════════════════════════════════════════

def _hf_text2img(prompt: str, model: str = FAST_MODEL) -> Image.Image | None:
    if not HF_TOKEN:
        log.error("HF_TOKEN not set in .env")
        return None

    url = f"{HF_BASE_URL}/{model}"
    payload = {
        "inputs": prompt,
        "parameters": TURBO_PARAMS,
        "options": {"wait_for_model": True},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(f"[HF] Attempt {attempt} | {model}")
            resp = requests.post(url, headers=HF_HEADERS, json=payload, timeout=TIMEOUT)

            if resp.status_code == 200:
                log.info("[HF] ✅ Success")
                return Image.open(BytesIO(resp.content)).convert("RGB")
            elif resp.status_code == 503:
                log.warning(f"[HF] Model loading, waiting {RETRY_SLEEP}s…")
                time.sleep(RETRY_SLEEP)
            elif resp.status_code == 401:
                log.error("[HF] ❌ Invalid HF_TOKEN")
                return None
            elif resp.status_code == 429:
                log.warning("[HF] Rate limited, waiting…")
                time.sleep(RETRY_SLEEP * 2)
            else:
                log.error(f"[HF] Status {resp.status_code}: {resp.text[:200]}")
                time.sleep(RETRY_SLEEP)

        except requests.exceptions.Timeout:
            log.warning(f"[HF] Timeout on attempt {attempt}")
        except Exception as e:
            log.error(f"[HF] Error: {e}")
            return None

    return None


def _hf_img2img(prompt: str, init_image: Image.Image, strength: float = 0.6) -> Image.Image | None:
    if not HF_TOKEN:
        return None

    url = f"{HF_BASE_URL}/{INPAINT_MODEL}"
    init_resized = init_image.resize((512, 512), Image.LANCZOS)
    buf = BytesIO()
    init_resized.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    payload = {
        "inputs": prompt,
        "parameters": {
            "init_image": img_b64,
            "strength": strength,
            "num_inference_steps": 20,
            "guidance_scale": 7.5,
        },
        "options": {"wait_for_model": True},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=HF_HEADERS, json=payload, timeout=60)
            if resp.status_code == 200:
                return Image.open(BytesIO(resp.content)).convert("RGB")
            elif resp.status_code == 503:
                time.sleep(RETRY_SLEEP * attempt)
            else:
                log.error(f"[img2img] {resp.status_code}: {resp.text[:200]}")
                time.sleep(RETRY_SLEEP)
        except Exception as e:
            log.error(f"[img2img] Error attempt {attempt}: {e}")

    return None


# ═════════════════════════════════════════════════════════════════════════════
# 1. OUTFIT IMAGES  — generated in PARALLEL
# ═════════════════════════════════════════════════════════════════════════════

def _build_outfit_prompt(desc: str, style_context: str) -> str:
    return (
        f"fashion model wearing {desc}, {style_context}, "
        "full body, white background, front view, fashion photography"
    )


def generate_outfit_images(
    outfit_descriptions: list[str],
    style_context: str = "",
) -> list[dict]:
    """
    Generate outfit images in PARALLEL — all start at the same time.
    Returns list of dicts: {prompt, image, base64}
    """
    prompts = [_build_outfit_prompt(d, style_context) for d in outfit_descriptions]

    # Parallel calls — 3 images generate simultaneously instead of sequentially
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        future_to_idx = {
            executor.submit(_hf_text2img, prompt): i
            for i, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            i = future_to_idx[future]
            img = future.result()
            results[i] = {
                "prompt": outfit_descriptions[i],
                "image": img,
                "base64": _pil_to_base64(img) if img else None,
            }

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 2. PINTEREST INSPO  — Unsplash (instant) + AI in parallel
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_unsplash(query: str, count: int = 3) -> list[Image.Image]:
    if not UNSPLASH_KEY:
        return []
    try:
        resp = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": count, "orientation": "portrait",
                    "client_id": UNSPLASH_KEY},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        images = []
        for photo in resp.json().get("results", []):
            img_url = photo["urls"].get("small")   # small = faster than regular
            r = requests.get(img_url, timeout=10)
            if r.status_code == 200:
                images.append(Image.open(BytesIO(r.content)).convert("RGB"))
        log.info(f"[Unsplash] ✅ {len(images)} photos for '{query}'")
        return images
    except Exception as e:
        log.warning(f"[Unsplash] {e}")
        return []


def generate_pinterest_inspo(
    style_keywords: list[str],
    n_real: int = 2,
    n_generated: int = 1,
) -> list[dict]:
    """
    Fetch Unsplash photos (fast, instant) + AI-generated inspo in parallel.
    Unsplash images appear immediately; AI images fill remaining slots.
    """
    results = []

    for keyword in style_keywords:
        # Unsplash fetch is fast — do first
        real_imgs = _fetch_unsplash(f"{keyword} fashion outfit", count=n_real)
        for img in real_imgs:
            results.append({"source": "unsplash", "keyword": keyword,
                            "image": img, "base64": _pil_to_base64(img)})

        # AI images only if Unsplash didn't fill slots
        n_gen = max(0, (n_real + n_generated) - len(real_imgs))
        if n_gen > 0:
            prompt = f"Pinterest fashion inspo, {keyword}, editorial aesthetic, studio"
            img = _hf_text2img(prompt)
            if img:
                results.append({"source": "generated", "keyword": keyword,
                                "image": img, "base64": _pil_to_base64(img)})

    return results


# ═════════════════════════════════════════════════════════════════════════════
# 3. VIRTUAL TRY-ON
# ═════════════════════════════════════════════════════════════════════════════

def _composite_overlay(user_photo: Image.Image, outfit_img: Image.Image) -> Image.Image:
    """Blend outfit onto lower body of user photo as fallback."""
    user = user_photo.convert("RGBA").copy()
    w, h = user.size
    outfit_h = int(h * 0.70)
    outfit_resized = outfit_img.resize((w, outfit_h), Image.LANCZOS).convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    overlay.paste(outfit_resized, (0, int(h * 0.30)))
    return Image.blend(user, overlay, alpha=0.45).convert("RGB")


def virtual_tryon(
    user_photo: Image.Image,
    outfit_description: str,
    hair_makeup_description: str = "",
    accessories: str = "",
    use_ai_compositing: bool = True,
) -> dict:
    """
    Virtual try-on: overlays recommended outfit + hair/makeup onto user photo.
    Tries AI compositing first, falls back to blend overlay.
    """
    if not isinstance(user_photo, Image.Image):
        return {"tryon_image": None, "base64": None, "success": False}

    user_resized = _resize_keep_aspect(user_photo, max_side=512)
    full_style = ", ".join(filter(None, [outfit_description, hair_makeup_description, accessories]))
    tryon_prompt = f"person wearing {full_style}, full body, fashion editorial, realistic"

    tryon_result = None
    method_used = "none"

    # Try AI img2img first
    if use_ai_compositing and HF_TOKEN:
        tryon_result = _hf_img2img(tryon_prompt, init_image=user_resized, strength=0.55)
        if tryon_result:
            method_used = "ai_img2img"

    # Fallback: generate outfit + blend
    if tryon_result is None:
        outfit_img = _hf_text2img(
            f"fashion model wearing {full_style}, white background, full body"
        )
        if outfit_img:
            tryon_result = _composite_overlay(user_resized, outfit_img)
            method_used = "blend_fallback"

    final_img = tryon_result
    return {
        "tryon_image": final_img,
        "base64": _pil_to_base64(final_img) if final_img else None,
        "method": method_used,
        "success": final_img is not None,
    }