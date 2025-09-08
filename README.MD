## AI Object Counting – Full Stack (Backend + Frontend)

This repository contains a Flask backend that orchestrates an AI pipeline (SAM + Transformers) and a Vite/React frontend. It supports single‑image counting, batch processing, user feedback, history, and performance metrics.

The project is containerized with Docker for an easy local setup.

---

## Quick Start (Docker Compose)

Prerequisites: Docker + Docker Compose installed.

1) Build and start services

```
docker compose up --build
```

Services:
- Backend API: http://localhost:5000
- Frontend UI: http://localhost:3000

By default, the backend runs in production mode with MySQL (a MySQL container is started by compose). Uploaded images and logs are mounted to local folders.

2) Stop and remove containers

```
docker compose down
```

### Using SQLite instead of MySQL (simple local run)

If you prefer not to run MySQL, you can run the backend locally in development (SQLite):

```
python start_development.py

cd seth_front_end
npm install
npm run dev
```

Then open the frontend at http://localhost:3000 (Vite proxy will forward /api to http://localhost:5000).

---

## Environment Variables (Docker)

Backend (see docker-compose.yml):
- OBJ_DETECT_ENV=production
- DATABASE_TYPE=mysql
- MYSQL_HOST=mysql
- MYSQL_PORT=3306
- MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
- SECRET_KEY
- LOG_LEVEL=INFO

Frontend:
- VITE_API_BASE_URL=http://localhost:5000
  - The frontend calls the backend through this URL.

Volumes:
- ./media:/app/media (uploaded images)
- ./logs:/app/logs
- MySQL data persisted in a named volume.

---

## API Endpoints (excerpt)

- Health: `GET /health`
- Object types: `GET /api/object-types`
- Single image count: `POST /api/count` (form‑data: image, object_type, [description])
- Auto‑detect all objects: `POST /api/count-all`
- Batch processing: `POST /api/batch/process`
- Results list (paged): `GET /api/results?page=1&per_page=20[&object_type=person]`
- Result details: `GET /api/results/{id}`
- Update correction: `PUT /api/correct/{id}` (json: { corrected_count })
- Delete result: `DELETE /api/results/{id}`

---

## Local Development (without Docker)

Backend:
```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python start_development.py
```

Frontend:
```
cd seth_front_end
npm install
npm run dev
```

---

## Project Structure

- `src/` – Flask app, API routes, storage engine, pipeline
- `seth_front_end/` – Vite/React app (TypeScript)
- `Dockerfile` – Backend container (Python)
- `seth_front_end/Dockerfile` – Frontend container (Node → static nginx)
- `docker-compose.yml` – Orchestration of backend, frontend, MySQL
- `media/` – Uploaded images (mounted volume in Docker)
- `logs/` – Logs (mounted volume in Docker)

---

## Notes

- The AI models (SAM + Transformers) download on first use and can be large; first processing takes longer.
- For production, set proper secrets (SECRET_KEY, database credentials) and consider a reverse proxy/SSL.
- CORS: backend exposes /api/* with permissive CORS for local usage.
