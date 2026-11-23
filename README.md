<div align="center">

# 🌾 AgriVision AI https://agrivision-ai-five.vercel.app/

### 3-Tier Hybrid Universal AI Crop Disease Detection for Indian Farmers

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![Supabase](https://img.shields.io/badge/Supabase-Database-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-Frontend-000000?style=for-the-badge&logo=vercel&logoColor=white)
![Render](https://img.shields.io/badge/Render-Backend-46E3B7?style=for-the-badge&logo=render&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

Upload a crop image → Get AI-assisted disease screening and treatment information in **English, Hindi, and Marathi**. Confirm diagnoses visually; predictions are not a substitute for expert advice.

[Features](#-features) · [Architecture](#-architecture) · [Tech Stack](#-tech-stack) · [Quick Start](#-quick-start) · [API Docs](#-api-documentation) · [Deployment](#-deployment)

</div>

---

## 🎯 Problem

Indian agriculture loses **₹90,000+ crores annually** to crop diseases. Farmers in rural areas lack access to timely diagnosis and expert advice. AgriVision AI solves this by putting the power of state-of-the-art cloud and offline edge Artificial Intelligence in every farmer's pocket.

## 💡 Solution: The 3-Tier AI Cascade

AgriVision AI uses a unique, resilient **three-tiered cascade architecture** to diagnose crop leaves:

1. **Tier 1 (Gemini 3.8 Flash Vision API)**: Attempts cloud image analysis when configured. Crop and disease results are not guaranteed and should be checked against the image.
2. **Tier 2 (OpenRouter Vision API Fallback)**: Tries a second cloud vision provider if Gemini is unavailable or conflicts with a supplied crop hint.
3. **Tier 3 (Local TF MobileNetV2 Classifier)**: If both cloud providers fail, local inference is available only when the user explicitly selects a leaf image. It uses 46 mapped leaf-disease classes across 15 crops, rejects low-confidence predictions, and filters to a supplied crop hint. Fruit, other, and unspecified images are declined rather than forced into a leaf label. Accuracy has not been independently re-evaluated because the labeled dataset is not included.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Disease Detection** | Upload a plant photo for cloud-assisted diagnosis; local fallback is restricted to selected leaf images |
| 📊 **Confidence Scoring** | Visual metrics showing diagnosis reliability |
| ⚠️ **Severity Assessment** | High / Moderate / Low severity classification |
| 💊 **Treatment Advice** | Organic and chemical treatment recommendations |
| 🛡️ **Prevention Tips** | Actionable steps to prevent disease spread |
| 🗣️ **Multilingual Support** | Full UI and translation in English, Hindi, and Marathi |
| 👤 **User Profile Page** | Unified profile details (Name, Email, Location) with personal analytics |
| 📈 **Analytics Dashboard** | Premium charts displaying scan trends, crop distribution, and severity donut |
| 📜 **Clickable History** | View previous diagnostic records inside animated modal popovers |
| 🚫 **Wildcard Routing** | Themed 404 page ("Page Not Found") for invalid URL fallbacks |
| 🐳 **Docker Ready** | One-command local containerization with Docker Compose |

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│           React Frontend (Vite + Tailwind)             │
│   Upload UI · Visual Dashboard · History · Profile     │
└───────────────────────────┬────────────────────────────┘
                            │ REST API (HTTPS + JWT)
┌───────────────────────────┴────────────────────────────┐
│                    FastAPI Backend                     │
│      Routers · Schemas · Security Headers · Auth       │
├───────────────────────────┬────────────────────────────┤
│   Universal AI Engine     │  Local Classifier (TF)     │
│   Gemini 3.8 / OpenRouter │  MobileNetV2 (Offline)     │
└───────────────────────────┼────────────────────────────┘
                            │ DB Clients (RLS Enabled)
┌───────────────────────────┴────────────────────────────┐
│             Supabase Cloud Infrastructure             │
│     User Database · Blob Storage · Auth Management     │
└────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **AI/ML** | Gemini 3.8 Flash, TensorFlow 2.x, MobileNetV2 | Hybrid vision APIs & offline edge classifier |
| **Backend** | FastAPI, Python 3.11, Pydantic, Uvicorn | Async REST API with automatic security headers |
| **Frontend** | React 18, Vite, Tailwind CSS, Framer Motion | Animated, responsive modern user experience |
| **Charts** | Recharts | Premium custom gradient bar charts & donut distributions |
| **Database** | Supabase (PostgreSQL) | User database records guarded by Row-Level Security (RLS) |
| **Storage** | Supabase Storage | Secure blob bucket storage for uploaded leaf snapshots |
| **Auth** | JWT + bcrypt | Secure session management & route authentication |
| **Container** | Docker, Docker Compose | Consistent local and staging docker runs |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn

### 1. Clone the repository
```bash
git clone https://github.com/SuhelMulla22/agrivision-ai.git
cd agrivision-ai
```

### 2. Start the Backend
```bash
cd backend
# Create virtual environment and install packages
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Linux/macOS
pip install -r requirements.txt -r requirements-ml.txt

# Copy the credential-free local demo configuration (uses the bundled model)
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
> API docs available at: http://127.0.0.1:8000/docs

Install `requirements-dev.txt` as well to run the backend test suite. Vercel
uses only `requirements.txt`; the local TensorFlow model dependencies are
isolated in `requirements-ml.txt`.

### 3. Start the Frontend
```bash
cd frontend
# Install dependencies
npm install

# Create frontend/.env from frontend/.env.example and set the Supabase project URL
# and publishable/anon key as VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.
# Start Vite server (runs on port 5176 to avoid local caching conflicts)
npm run dev
```
> App available at: http://localhost:5176

### Supabase Authentication

Email/password registration, email confirmation, Google OAuth, password reset,
and API access-token validation use Supabase Auth. Configure the Supabase project
URL and **publishable/anon** key in both `backend/.env` and `frontend/.env`.
For server-side prediction history and image storage, configure
`SUPABASE_SERVICE_ROLE_KEY` in the backend only. Never put a Supabase
`service_role` key in the frontend. Run
[`backend/supabase/migrations/001_initial_schema.sql`](./backend/supabase/migrations/001_initial_schema.sql)
in the Supabase SQL Editor to create the predictions table and `crop-images`
storage bucket.

In Supabase Authentication → URL Configuration, set the local Site URL to
`http://localhost:5176` and allow `http://localhost:5176/auth/callback`.
Add your deployed app origin and `/auth/callback` there as well. In Google Cloud,
the OAuth redirect URI is the Supabase callback URL shown by the Google provider
settings (usually `https://<project-ref>.supabase.co/auth/v1/callback`); it is
not the app callback URL. Restart Vite after changing frontend environment
variables.

---

## 📡 API Documentation

Interactive API documents are available at:

| Docs | URL |
|------|-----|
| **Swagger UI** | http://127.0.0.1:8000/docs |
| **ReDoc** | http://127.0.0.1:8000/redoc |
| **OpenAPI JSON** | http://127.0.0.1:8000/openapi.json |

### Key Endpoints

```bash
# Health check
GET /api/health

# Predict disease (auth token required unless LOCAL_DEMO_MODE=true in development)
POST /api/v1/predict
  - file: image (multipart/form-data)
  - language: en | hi | mr

# Auth
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me

# History & Analytics (Requires auth token)
GET /api/v1/history
GET /api/v1/analytics
```

---

## ☁️ Deployment (Free Lifetime Setup)

Refer to the production configuration guidelines for hosting this stack at zero cost:

### Local Docker Compose

Copy `backend/.env.example` to `backend/.env`, set `SECRET_KEY` to a random secret
of at least 32 characters, and configure a cloud inference key if the local model
is unavailable. Set `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` in the
environment used by Docker Compose so they are available at frontend build time.
Then run `docker compose up --build`. The frontend is served at
http://localhost:3000 and proxies API requests to the backend.

### Vercel Services

The root `vercel.json` deploys the frontend and FastAPI backend as services in
one Vercel project. `/api/*` is routed publicly to the backend; all other paths
are routed publicly to the frontend. The browser calls the API through the
same-origin `/api` paths, so no service bindings are needed.

Configure these Vercel environment variables:

- Backend: `ENVIRONMENT=production`, a random `SECRET_KEY` of at least 32
  characters, `SKIP_TF_LOAD=true`, and at least one inference provider key
  (`GEMINI_API_KEY` or `OPENROUTER_API_KEY`). Set `SUPABASE_URL` and
  `SUPABASE_ANON_KEY` to enable Supabase authentication. Set
  `SUPABASE_SERVICE_ROLE_KEY` for server-side history and image persistence.
- Create `GEMINI_API_KEY` in Google AI Studio. It is not the Google OAuth
  client ID or client secret used for Supabase sign-in.
- Run `backend/supabase/migrations/001_initial_schema.sql` in Supabase's SQL
  Editor to create the history table and storage bucket.
- Frontend build: `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY`.
- Leave `VITE_API_URL` unset or empty; the frontend uses same-origin `/api`
  requests through the public rewrite.

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Suhel Mulla** — B.Tech AI/ML, D Y Patil Agriculture and Technical University, Talsande

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/suhelmulla22)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SuhelMulla22)

---

<div align="center">

**Built with 🌱 for Indian Agriculture**

*AgriVision AI — Because every farmer deserves access to expert plant pathology.*

</div>
