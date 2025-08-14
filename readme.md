# Indian Language Translator

A web-based translation service that translates English text to various Indian languages using the IndicTrans2 model.

## Features

### Web Interface
- **Text Input**: Large text area for entering English text (up to 500 characters)
- **Language Selection**: Dropdown menu to select target Indian language
- **Supported Languages**:
  - Hindi (हिंदी)
  - Tamil (தமிழ்)
  - Telugu (తెలుగు)
  - Kannada (ಕನ್ನಡ)
  - Malayalam (മലയാളം)
  - Bengali (বাংলা)
  - Marathi (मराठी)
  - Gujarati (ગુજરાતી)
  - Odia (ଓଡ଼ିଆ)
  - Punjabi (ਪੰਜਾਬੀ)
- **Real-time Translation**: Click translate button to get instant results
- **Copy to Clipboard**: Easy copying of translated text
- **Responsive Design**: Works on desktop and mobile devices

#### PMIndia dataset has been used already to train IndicTrans2 Model Hence doesn't need separate re-training
> "The NMT model used in our system (IndicTrans2) was pre-trained on multiple Indian language corpora, including the PMIndia dataset — a publicly available parallel corpus of Prime Minister’s speeches in multiple Indian languages (Haddow & Kirefu, 2020)."

Haddow, B., & Kirefu, F. (2020). PMIndia: A Parallel Corpus of the Prime Minister of India's Speeches. https://github.com/bhaddow/pmindia-crawler


### API Endpoints
- `GET /` - Web interface
- `POST /api/v1/translate` - Translation API
- `GET /health` - Health check
- `GET /docs` - API Docs (swagger)

## Usage

### Running Locally
```bash
pip install -r requirements.txt
python main.py
```

### Using Docker
```bash
docker build -t translator .
docker run -p 8000:8000 translator
```

### Access the Application
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## API Usage

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
     -H "Content-Type: application/json" \
     -d '{"source_text": "Hello, how are you?", "target_lang": "hi"}'
```

## Technologies Used
- FastAPI for web framework
- IndicTrans2 model for translation
- Bootstrap for UI design
- Jinja2 for templating