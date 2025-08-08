import os
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

# Set device for torch (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        print("Starting up... Loading model and tokenizer")
        get_tokenizer()
        get_model()
        print("Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"Warning: Could not pre-load model during startup: {e}")
        print("Model will be loaded on first request")
    yield
    # Shutdown
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Allow all origins (for development; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Language tag mapping: ISO 639-1 → IndicTrans2 format
LANGUAGE_TAGS = {
    "hi": "hin_Deva",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "ml": "mal_Mlym",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "or": "ory_Orya",
    "pa": "pan_Guru"
}

# ✅ Load model and tokenizer with LRU caching
@lru_cache(maxsize=1)
def load_tokenizer():
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "ai4bharat/indictrans2-en-indic-1B", 
            trust_remote_code=True,
            cache_dir="/app/.cache" if os.path.exists("/app") else None
        )
        print("Tokenizer loaded successfully")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise

@lru_cache(maxsize=1)
def load_model():
    try:
        print("Loading model...")
        
        # Try different loading strategies
        model_configs = [
            {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "cache_dir": "/app/.cache" if os.path.exists("/app") else None,
                "low_cpu_mem_usage": True
            },
            {
                "trust_remote_code": True,
                "torch_dtype": torch.float32,
                "cache_dir": "/app/.cache" if os.path.exists("/app") else None
            },
            {
                "trust_remote_code": True
            }
        ]
        
        model = None
        for i, config in enumerate(model_configs):
            try:
                print(f"Trying model loading strategy {i+1}...")
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    "ai4bharat/indictrans2-en-indic-1B",
                    **config
                )
                break
            except Exception as e:
                print(f"Strategy {i+1} failed: {e}")
                if i == len(model_configs) - 1:
                    raise
                continue
        
        if model is None:
            raise RuntimeError("Failed to load model with any strategy")
            
        model.eval()
        model.to(DEVICE)
        print(f"Model loaded successfully on {DEVICE}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Initialize tokenizer and model lazily
tokenizer = None
model = None

def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = load_tokenizer()
    return tokenizer

def get_model():
    global model
    if model is None:
        model = load_model()
    return model

# ✅ Request schema
class TranslationRequest(BaseModel):
    source_text: str
    target_lang: str  # 'hi', 'ta', etc.

# ✅ Cache translations (in-memory)
@lru_cache(maxsize=512)
def cached_translation(source_text: str, target_lang: str) -> str:
    if target_lang not in LANGUAGE_TAGS:
        raise ValueError("Unsupported language")
    try:
        tokenizer = get_tokenizer()
        model = get_model()
        
        src_tag = "eng_Latn"
        tgt_tag = LANGUAGE_TAGS[target_lang]
        tagged_text = f"{src_tag} {tgt_tag} {source_text.strip()}"
        
        inputs = tokenizer(tagged_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Clear any existing cache
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'clear_cache'):
                model.transformer.clear_cache()
            
            outputs = model.generate(
                **inputs, 
                max_length=512, 
                num_beams=1,  # Reduced beam size to avoid cache issues
                early_stopping=True,
                do_sample=False,
                use_cache=False,  # Disable cache to avoid the error
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return translated.strip()
    except Exception as e:
        print(f"Translation error: {e}")
        raise RuntimeError(f"Model translation failed: {e}")

# ✅ API route
@app.post("/api/v1/translate")
def translate(request: TranslationRequest):
    try:
        translated_text = cached_translation(request.source_text, request.target_lang)
        return {"translated_text": translated_text}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))

# ✅ Web interface route
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ✅ Health check endpoint
@app.get("/health")
def health_check():
    try:
        # Try to get tokenizer and model to verify they're working
        tokenizer = get_tokenizer()
        model = get_model()
        return {
            "status": "healthy", 
            "device": str(DEVICE),
            "model_loaded": model is not None,
            "tokenizer_loaded": tokenizer is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "device": str(DEVICE)
        }

# ✅ Start app using Azure-provided PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Azure will set this
    uvicorn.run(app, host="0.0.0.0", port=port)
