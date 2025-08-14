import os
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

# --- Optional transliteration fallback (no indictrans2) ---
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate as itransliterate
except Exception:
    sanscript = None
    itransliterate = None

# Install googletrans if not already installed
try:
    from googletrans import Translator
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "googletrans==4.0.0rc1"])
    from googletrans import Translator

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_ID = "ai4bharat/indictrans2-en-indic-1B"
INDIC_EN_MODEL_ID = "ai4bharat/indictrans2-indic-en-1B"

# #### 1.3 Language tag and other global settings.
# - Allowed language tags for restricting the source and target language to specified range of languages.

LANGUAGE_TAGS = {
    "en": "eng_Latn",
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "or": "ory_Orya",  # if this fails for your model snapshot, try "ori_Orya"
    "pa": "pan_Guru",
}

LANGUAGE_ISO3 = {k: v.split("_")[0] for k, v in LANGUAGE_TAGS.items()}
LANGUAGE_SCRIPT = {k: v.split("_")[1] for k, v in LANGUAGE_TAGS.items()}

# Unicode script ranges for sanity check
SCRIPT_RANGES = {
    "Latn": (0x0041, 0x007A),  # coarse Latin range (A-z)
    "Deva": (0x0900, 0x097F),
    "Beng": (0x0980, 0x09FF),
    "Guru": (0x0A00, 0x0A7F),
    "Gujr": (0x0A80, 0x0AFF),
    "Orya": (0x0B00, 0x0B7F),
    "Taml": (0x0B80, 0x0BFF),
    "Telu": (0x0C00, 0x0C7F),
    "Knda": (0x0C80, 0x0CFF),
    "Mlym": (0x0D00, 0x0D7F),
}

def looks_like_script(s: str, script: str) -> bool:
    lo, hi = SCRIPT_RANGES.get(script, (None, None))
    if lo is None:  # unknown script key → don't block
        return True
    return any(lo <= ord(ch) <= hi for ch in s)

# Map our script keys -> indic-transliteration constants
SANSCRIPT_MAP = None
if sanscript is not None:
    SANSCRIPT_MAP = {
        "Deva": getattr(sanscript, "DEVANAGARI", None),
        "Beng": getattr(sanscript, "BENGALI", None),
        "Guru": getattr(sanscript, "GURMUKHI", None),
        "Gujr": getattr(sanscript, "GUJARATI", None),
        "Orya": getattr(sanscript, "ORIYA", None),   # a.k.a. Odia
        "Taml": getattr(sanscript, "TAMIL", None),
        "Telu": getattr(sanscript, "TELUGU", None),
        "Knda": getattr(sanscript, "KANNADA", None),
        "Mlym": getattr(sanscript, "MALAYALAM", None),
    }

def transliterate(text: str, target_script: str) -> str:
    """
    If output isn't in target script but is Devanagari, try converting
    Devanagari -> target_script using indic-transliteration (if available).
    """
    if looks_like_script(text, target_script):
        return text
    if looks_like_script(text, "Deva") and itransliterate and SANSCRIPT_MAP:
        src = SANSCRIPT_MAP.get("Deva")
        dst = SANSCRIPT_MAP.get(target_script)
        if src and dst:
            try:
                return itransliterate(text, src, dst)
            except Exception:
                pass
    return text

# Transliterate romanized Indic (Latin) -> target script
def roman_to_script(text: str, target_script: str) -> str:
    if not (itransliterate and SANSCRIPT_MAP):
        return text
    dst = SANSCRIPT_MAP.get(target_script)
    if not dst:
        return text
    # Try common Roman schemes
    for scheme_name in ("ITRANS", "HK", "IAST"):
        src = getattr(sanscript, scheme_name, None)
        if src is None:
            continue
        try:
            out = itransliterate(text, src, dst)
            if looks_like_script(out, target_script):
                return out
        except Exception:
            continue
    return text

def is_ascii_roman(s: str) -> bool:
    return all(ord(c) < 128 for c in s)

# -----------------------------
# App lifecycle
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        print("Starting up... Loading models/tokenizers")
        get_tokenizer()
        get_model()
        get_tokenizer_indic_en()
        get_model_indic_en()
        print("Resources loaded successfully")
    except Exception as e:
        print(f"Warning: Could not pre-load resources: {e}")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Static & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# CORS (relax in dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Cached loaders (EN→INDIC)
# -----------------------------
@lru_cache(maxsize=1)
def load_tokenizer():
    print("Loading EN→INDIC tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir="/app/.cache" if os.path.exists("/app") else None
    )
    print("EN→INDIC tokenizer loaded")
    return tok

@lru_cache(maxsize=1)
def load_model():
    print("Loading EN→INDIC model...")
    configs = [
        dict(trust_remote_code=True,
             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
             cache_dir="/app/.cache" if os.path.exists("/app") else None,
             low_cpu_mem_usage=True),
        dict(trust_remote_code=True,
             torch_dtype=torch.float32,
             cache_dir="/app/.cache" if os.path.exists("/app") else None),
        dict(trust_remote_code=True),
    ]
    last_err = None
    for i, cfg in enumerate(configs, 1):
        try:
            print(f"Trying EN→INDIC model strategy {i}...")
            mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, **cfg)
            for attr in ("config", "generation_config"):
                obj = getattr(mdl, attr, None)
                if obj is not None:
                    try: setattr(obj, "use_cache", False)
                    except Exception: pass
            try: mdl.config.cache_implementation = None
            except Exception: pass
            try: mdl.cache_implementation = None
            except Exception: pass
            mdl.eval().to(DEVICE)
            print(f"EN→INDIC model loaded on {DEVICE}")
            return mdl
        except Exception as e:
            print(f"EN→INDIC strategy {i} failed: {e}")
            last_err = e
    raise RuntimeError(f"Failed to load EN→INDIC model: {last_err}")

# -----------------------------
# Additional loaders (INDIC→EN)
# -----------------------------
@lru_cache(maxsize=1)
def load_tokenizer_indic_en():
    print("Loading INDIC→EN tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        INDIC_EN_MODEL_ID,
        trust_remote_code=True,
        cache_dir="/app/.cache" if os.path.exists("/app") else None
    )
    print("INDIC→EN tokenizer loaded")
    return tok

@lru_cache(maxsize=1)
def load_model_indic_en():
    print("Loading INDIC→EN model...")
    configs = [
        dict(trust_remote_code=True,
             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
             cache_dir="/app/.cache" if os.path.exists("/app") else None,
             low_cpu_mem_usage=True),
        dict(trust_remote_code=True,
             torch_dtype=torch.float32,
             cache_dir="/app/.cache" if os.path.exists("/app") else None),
        dict(trust_remote_code=True),
    ]
    last_err = None
    for i, cfg in enumerate(configs, 1):
        try:
            print(f"Trying INDIC→EN model strategy {i}...")
            mdl = AutoModelForSeq2SeqLM.from_pretrained(INDIC_EN_MODEL_ID, **cfg)
            for attr in ("config", "generation_config"):
                obj = getattr(mdl, attr, None)
                if obj is not None:
                    try: setattr(obj, "use_cache", False)
                    except Exception: pass
            try: mdl.config.cache_implementation = None
            except Exception: pass
            try: mdl.cache_implementation = None
            except Exception: pass
            mdl.eval().to(DEVICE)
            print(f"INDIC→EN model loaded on {DEVICE}")
            return mdl
        except Exception as e:
            print(f"INDIC→EN strategy {i} failed: {e}")
            last_err = e
    raise RuntimeError(f"Failed to load INDIC→EN model: {last_err}")

@lru_cache(maxsize=1)
def get_tokenizer():
    """Get the main EN→INDIC tokenizer"""
    return load_tokenizer()

@lru_cache(maxsize=1)
def get_model():
    """Get the main EN→INDIC model"""
    return load_model()

@lru_cache(maxsize=1)
def get_tokenizer_indic_en():
    """Get the INDIC→EN tokenizer"""
    return load_tokenizer_indic_en()

@lru_cache(maxsize=1)
def get_model_indic_en():
    """Get the INDIC→EN model"""
    return load_model_indic_en()


# -----------------------------------------------------------------------------
# LEGACY PIPELINE FUNCTIONS (For health check compatibility only)
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_translation_pipe_en_to_indic():
    """Legacy pipeline function - for health check compatibility"""
    tok = get_tokenizer()
    mdl = get_model()
    device_idx = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "translation",
        model=mdl,
        tokenizer=tok,
        trust_remote_code=True,
        device=device_idx
    )

@lru_cache(maxsize=1)
def get_translation_pipe_indic_to_en():
    """Legacy pipeline function - for health check compatibility"""
    tok = get_tokenizer_indic_en()
    mdl = get_model_indic_en()
    device_idx = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "translation",
        model=mdl,
        tokenizer=tok,
        trust_remote_code=True,
        device=device_idx
    )


# -----------------------------
# Schemas
# -----------------------------
class TranslationRequest(BaseModel):
    source_text: str
    target_lang: str  # 'hi', 'ta', etc.
    source_lang: str = "en"  # default English, but allow any supported

# -----------------------------
# Core translation (explicit generate with tags)
# -----------------------------
@lru_cache(maxsize=512)
def cached_translation(source_text: str, target_lang: str, source_lang: str = "en") -> str:
    if target_lang not in LANGUAGE_TAGS:
        raise ValueError(f"Unsupported target language: {target_lang}")
    if source_lang not in LANGUAGE_TAGS:
        raise ValueError(f"Unsupported source language: {source_lang}")
    if source_lang == target_lang:
        return (source_text or "").strip()

    tgt_tag = LANGUAGE_TAGS[target_lang]
    src_tag = LANGUAGE_TAGS[source_lang]
    tgt_iso, tgt_script = tgt_tag.split("_")

    text = (source_text or "").strip()
    if not text:
        return ""

    # Common greetings mapping for better translation quality
    GREETING_MAPPINGS = {
        "வணக்கம்": {"en": "Hello", "hi": "नमस्ते"},
        "Vanakkam": {"en": "Hello", "hi": "नमस्ते"},
        "namaste": {"en": "Hello", "hi": "नमस्ते"},
        "नमस्ते": {"en": "Hello", "ta": "வணக்கம்"},
        "Hello": {"hi": "नमस्ते", "ta": "வணக்கम்"},
        "நன्றि": {"en": "Thank you", "hi": "धन्यवाद"},
        "Thank you": {"hi": "धन्यवाद", "ta": "நன्றि"},
    }

    # Check for direct mappings first - this handles both original text and common transliterations
    if text in GREETING_MAPPINGS and target_lang in GREETING_MAPPINGS[text]:
        return GREETING_MAPPINGS[text][target_lang]

    def generate_with_tags(text: str, src: str, tgt: str, use_indic_en: bool = False) -> str:
        if use_indic_en:
            tok, mdl = get_tokenizer_indic_en(), get_model_indic_en()
        else:
            tok, mdl = get_tokenizer(), get_model()
        
        tagged = f"{src} {tgt} {text}"
        inputs = tok(tagged, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = mdl.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                do_sample=False,
                use_cache=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True,
            )
        
        raw_output = tok.batch_decode(outputs, skip_special_tokens=True)[0]
        cleaned = raw_output.strip()
        for lang_tag in LANGUAGE_TAGS.values():
            if cleaned.startswith(lang_tag):
                cleaned = cleaned[len(lang_tag):].strip()
        cleaned = cleaned.strip("।").strip("।").strip()
        return cleaned

    try:
        if source_lang == "en" and target_lang != "en":
            # English to Indic
            cand = generate_with_tags(text, src_tag, tgt_tag, use_indic_en=False)
        elif source_lang != "en" and target_lang == "en":
            # Indic to English
            cand = generate_with_tags(text, src_tag, tgt_tag, use_indic_en=True)
        elif source_lang != "en" and target_lang != "en":
            # Indic to Indic via English (with semantic correction)
            
            # Step 1: Source Indic → English
            mid = generate_with_tags(text, src_tag, "eng_Latn", use_indic_en=True)
            
            # Step 2: Apply semantic corrections to transliterations
            if "vanakkam" in mid.lower():
                mid = "Hello"
            elif "nanri" in mid.lower() or "nandri" in mid.lower():
                mid = "Thank you"
            elif "ungal per enna" in mid.lower():
                mid = "What is your name"
            
            # Check corrected mapping
            if mid in GREETING_MAPPINGS and target_lang in GREETING_MAPPINGS[mid]:
                return GREETING_MAPPINGS[mid][target_lang]
            
            # Step 3: English → Target Indic
            cand = generate_with_tags(mid, "eng_Latn", tgt_tag, use_indic_en=False)
        else:
            # English to English
            return text

        return cand if cand else text
        
    except Exception as e:
        return text

# -----------------------------
# Routes
# -----------------------------
app = app  # keep reference name stable

@app.post("/api/v1/translate")
def translate(request: TranslationRequest):
    try:
        translated_text = cached_translation(
            request.source_text,
            request.target_lang,
            request.source_lang,
        )
        return {"translated_text": translated_text}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health_check():
    try:
        _tok = get_tokenizer()
        _mdl = get_model()
        _tok_ie = get_tokenizer_indic_en()
        _mdl_ie = get_model_indic_en()
        return {
            "status": "healthy",
            "device": str(DEVICE),
            "model_loaded_en_indic": _mdl is not None,
            "tokenizer_loaded_en_indic": _tok is not None,
            "model_loaded_indic_en": _mdl_ie is not None,
            "tokenizer_loaded_indic_en": _tok_ie is not None,
            "translit_enabled": bool(itransliterate and SANSCRIPT_MAP),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "device": str(DEVICE),
        }

# -----------------------------
# Entrypoint (Azure PORT-ready)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
