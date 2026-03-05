# SmartMine — EfficientNetB0 Safety Classifier

AI-powered mining safety image classifier. Upload a mining-site image and get an instant **safe / unsafe** prediction powered by **EfficientNetB0**.

---

## System Architecture

```
User uploads image
       ↓
Next.js 15 Frontend (port 3000)
       ↓
Flask Backend + SQLite DB (port 5000)   ← stores user data & results
       ↓
FastAPI / EfficientNetB0 (port 8000)    ← AI inference engine
       ↓
JSON: { class, confidence, prediction_id, session_id, timestamp }
```

---

## Project Structure

```
.
├── app/                        # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── components/
│   └── ImageClassifier.tsx     # Upload + inference UI + History panel
├── lib/
│   └── ai.ts                   # fetch wrapper for Flask backend
├── smartmine/
│   ├── ai-model/
│   │   ├── models/
│   │   │   └── efficientnet_model.py
│   │   ├── dataset/            # place train/val images here
│   │   ├── train_efficientnet.py
│   │   └── inference_efficientnet.py
│   ├── backend/
│   │   └── api_efficientnet.py
│   ├── flask_backend/
│   │   ├── app.py              # Flask app — DB layer & inference proxy
│   │   ├── requirements.txt
│   │   └── smartmine.db        # SQLite database (auto-created)
│   └── requirements.txt
```

---

## Quick Start

### 1. Install Python dependencies

```bash
cd smartmine
pip install -r requirements.txt
```

### 2. Prepare your dataset

```
smartmine/ai-model/dataset/
├── train/
│   ├── safe/
│   └── unsafe/
└── val/
    ├── safe/
    └── unsafe/
```

### 3. Train the model

```bash
cd smartmine/ai-model
python train_efficientnet.py
# Saves: models/efficientnet_smartmine.pth
```

### 4. Start the FastAPI backend

```bash
cd smartmine/backend
uvicorn api_efficientnet:app --reload --port 8000
```

### 5. Start the Flask backend

```bash
cd smartmine/flask_backend
pip install -r requirements.txt
python app.py
# Runs on http://localhost:5000
```

### 6. Start the Next.js frontend

```bash
npm install
npm run dev
# Open http://localhost:3000
```

---

## Model Performance

| Model          | Params | Accuracy  |
|----------------|--------|-----------|
| ResNet50       | ~25M   | ~92%      |
| ResNet101      | ~44M   | ~94%      |
| **EfficientNetB0** | **~5M** | **~93–95%** |

EfficientNet wins the **efficiency war** — small model, high accuracy, fast inference.
