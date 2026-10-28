# Deployment notes

Nothing in this repository has been deployed. Everything below is the plan the code is designed for; each step needs to be tried and verified in your environment before you rely on it.

## Shape of the system
- **Backend**: one FastAPI process (`scripts/run_server.py`, uvicorn). It runs one analysis at a time on a single worker thread with a bounded queue (`AVDD_MAX_PENDING`). It also serves the built frontend, so the simplest deployment is **one service, one origin** and no CORS.
- **Frontend**: static files from `frontend/dist` (`npm run build`). Host them on any static host only if you need a separate origin.

## Backend requirements
| Need | Detail |
|---|---|
| Python | 3.10+, with `pip install -e ".[video,hf,api]"` and CPU PyTorch (`pip install torch --index-url https://download.pytorch.org/whl/cpu`) |
| ffmpeg + ffprobe | on `PATH`. Required to read every media type |
| Model weights | The trained heads and YuNet ship in `src/avdd/assets` (<1 MB). The backbones `openai/clip-vit-large-patch14` (1.7 GB) and `facebook/wav2vec2-xls-r-300m` (1.2 GB) download from Hugging Face on first start and are cached; the host needs outbound HTTPS to huggingface.co once, or a pre-populated cache |
| Memory / CPU | About 4 GB RAM for both backbones on CPU. A 10 s video takes roughly 10-20 s on a 4-core CPU (12 face crops through ViT-L/14 dominate); a GPU (`AVDD_DEVICE=cuda`) makes it near-instant |
| Storage | `AVDD_DATA_DIR` holds the SQLite history and a temporary upload folder. Use a persistent disk if you want history to survive restarts. Uploads are deleted when each job ends and swept at startup |
| Limits | `AVDD_MAX_UPLOAD_MB` (100), `AVDD_MAX_DURATION_S` (60), `AVDD_MAX_IMAGE_MEGAPIXELS` (40). Raise the reverse proxy's body-size and timeout limits to match, or uploads will fail before reaching the app |

## Frontend on a separate origin
1. `cd frontend; $env:VITE_API_BASE = "https://api.example.org"; npm run build` and upload `dist/`.
2. Start the backend with `AVDD_CORS_ORIGINS=https://app.example.org`.
3. `VITE_API_BASE` is public. The backend has no API keys and the frontend contains no secrets.

## Privacy and security checklist (what the code does, and what it does not)
- The raw upload is streamed to a random temporary file, never named after the client's file, and deleted when the job ends (success or failure). Only the result JSON is stored.
- There is **no authentication, no per-user data separation and no rate limiting** beyond the queue bound. Anyone who can reach the server can upload files and read, export and delete every stored record. Put it behind an authenticating reverse proxy or a private network before exposing it.
- Put TLS in front of it. Do not claim enterprise-grade security, legal admissibility or forensic reliability; none has been established.
- Tell users that faces and voices are personal data and obtain consent before they upload other people's media.
- Disable the interactive API docs in production with `AVDD_ENABLE_DOCS=0`.

## Verifying a deployment
`GET /api/health` (process up), `GET /api/ready` (models loaded; 503 while loading), `GET /api/capabilities` (limits and per-model state), then upload one file of each kind from the UI.
