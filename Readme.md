# Football Player Heatmap & Tactical Analytics Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![React Version](https://img.shields.io/badge/react-18%2B-blue.svg)](https://react.dev/)
[![Docker Compose](https://img.shields.io/badge/docker%20compose-supported-blue.svg)](https://docs.docker.com/compose/)
[![CI Status](https://img.shields.io/badge/CI-active-green.svg)](https://github.com/)

A comprehensive full-stack computer vision analytics platform designed for broadcast football video footage. The application extracts frames, detects and tracks players across the match, classifies players into teams by jersey color, maps coordinates to real pitch dimensions via homography calibration, and generates interactive heatmaps, trajectories, and detailed tactical analytics.

---

## 🌟 Key Features

- **End-to-End CV Pipeline:** Runs object detection (YOLO) and multi-object tracking (ByteTrack-style Kalman filter + Hungarian IoU matching) to resolve occlusions.
- **Automatic Team Classification:** Performs green/grass extraction using HSV masking, torso cropping, and LAB color space K-Means clustering (K=3 to isolate referee kits).
- **Homography Mapping & Tactical Analytics:** Translates screen pixels to real-world pitch coordinates (meters) using OpenCV homography. Computes metrics like speed, distance, acceleration, direction, and team spacing (compactness, width, depth).
- **Interactive Visualizations:** Generates Gaussian Kernel Density Estimation (KDE) heatmaps, smoothed trajectories (via Savitzky-Golay filtering), and telemetry overlays.
- **RESTful API:** Robust FastAPI backend with background job orchestration and detailed schemas.
- **Modern Dashboard:** Interactive dashboard built in React + TypeScript + Tailwind CSS, showing match, team, and player-level insights.
- **Reports Export:** Export match reports into JSON, CSV, or high-fidelity PDF formats.

---

## 🏗️ Architecture

```
football-analytics-platform/
├── .github/                 # GitHub workflows & templates
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/             # API routes (matches, dashboard, export)
│   │   ├── core/            # Config loaded from env vars
│   │   ├── db/              # DB schemas, models, and sessions
│   │   ├── cv_pipeline/     # Detection, tracking, team clustering, homography
│   │   ├── schemas/         # Pydantic contract validators
│   │   ├── services/        # Pipeline orchestration (background workers)
│   │   └── utils/           # Video processing and I/O helpers
│   ├── requirements.txt     # Python requirements
│   └── Dockerfile
├── frontend/                # Vite + React + TS + Tailwind frontend
│   ├── src/                 # Pages, components, api clients, assets
│   ├── index.html
│   ├── tsconfig.json
│   └── Dockerfile
├── docker-compose.yml       # Production-ready Compose orchestration
├── LICENSE                  # MIT License
└── README.md                # Project documentation
```

---

## ⚙️ VisionX OD-17 Computer Vision & Tracking Architecture

```
Frame Extraction (OpenCV)
  │
  ▼
Player Detection (YOLO, via Ultralytics)
  │
  ▼
Player Filtering (PlayerValidator: Class, Aspect Ratio [1.1-4.5], Area Ratio [0.01%-15%], ROI Pitch Filter)
  │  └── Rejects Drones, Sky Cams, Banners, and Non-Player Objects
  ▼
Multi-Cue Feature Extraction (Domain 3 Descriptors)
  ├── 1. Color Features: HSV Hue Histogram + Torso LAB Mean Color (Grass Masked Out)
  ├── 2. Shape Features: Bounding Box Geometry + 7 Invariant Hu Moments
  ├── 3. Texture Features: Local Binary Pattern (LBP 8-neighbor 10-bin Histogram)
  ├── 4. Edge Features: Canny Edge Density & 4-Quadrant Spatial Distribution
  ├── 5. Geometric Features: Homography Pitch Coordinates (X_m, Y_m)
  └── 6. Landmark Features: 5-Point Normalized Body Keypoints
  │
  ▼
Motion Estimation (8-State Constant-Velocity Kalman Filter Prediction)
  │
  ▼
Multi-Cue Hungarian Association (Data Association)
  └── TOTAL_COST = α * IoU_Cost + β * Appearance_Cost + γ * Motion_Cost + δ * Geometry_Cost
  │
  ▼
Track Lifecycle Management (TENTATIVE ➔ CONFIRMED ➔ LOST ➔ REMOVED)
  └── Requires MIN_CONFIRMATION_HITS (3) before assigning persistent Player ID
  │
  ▼
Feature Gallery Re-Identification (ReID Recovery)
  └── Searches Lost-Track Gallery + Speed Plausibility Filter to restore IDs after occlusions/crossing
  │
  ▼
Persistent Player Tracking IDs & Event Debug Log
  │
  ▼
Trajectory & Gaussian Heatmap Analytics Output
```

---

## 🛠️ Honest Scope Notes

*   **Detection Weights:** Uses the generic COCO-pretrained `yolo11n.pt` model ("person" class) out of the box. To improve class precision (distinguishing players, referees, and the ball), fine-tune on a custom dataset (e.g., SoccerNet-Tracking) and update `YOLO_MODEL_PATH` in `.env`.
*   **Homography Calibration:** A static homography matrix is assumed. Standard broadcast cameras pan and zoom. You can calibrate specific frames via the `POST /matches/{id}/calibrate` API endpoint using a 4-point calibration. Without it, the backend uses a coarse full-frame-to-pitch fallback.
*   **Tracking Accuracy:** The `avg_tracking_accuracy` is a confidence-derived heuristic proxy rather than a ground-truth MOTA/IDF1 metric.
*   **Player Thumbnails:** Bounding-box crops are used as team/player profile indicators.

---

## 🚀 Running the Project

### 🐳 With Docker (Recommended)

Make sure you have Docker and Docker Compose installed.

```bash
docker compose up --build
```

- **Frontend Dashboard:** [http://localhost:5173](http://localhost:5173)
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **PostgreSQL Database:** `localhost:5432`

*Note: The first run will automatically download YOLOv11 weights, which might take a few minutes.*

---

### 💻 Running Locally

#### 🐍 Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows (PowerShell):
   .\venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template and adjust database credentials if needed:
   ```bash
   cp .env.example .env
   ```
5. Run the database setup script (requires Postgres running locally):
   ```bash
   python create_db.py
   ```
6. Start the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

#### ⚛️ Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```
2. Install npm packages:
   ```bash
   npm install
   ```
3. Start the Vite development server:
   ```bash
   npm run dev
   ```

---

## 🔌 API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **POST** | `/api/v1/matches/upload` | Upload a new match video and initialize match records. |
| **POST** | `/api/v1/matches/{id}/calibrate` | Calibrate coordinate homography matrix using 4 points. |
| **POST** | `/api/v1/matches/{id}/analyze` | Start running the background CV pipeline on the uploaded video. |
| **GET** | `/api/v1/matches/{id}/status` | Poll CV job status and percent progress. |
| **GET** | `/api/v1/matches/{id}/analytics` | Get match-level results, statistics, and ball tracking metrics. |
| **GET** | `/api/v1/matches/{id}/heatmap` | Fetch match-level Gaussian KDE density matrix. |
| **GET** | `/api/v1/matches/{id}/consistency-log` | Real-time Track Consistency Monitor: inspect track assignments, ReID recoveries, and ID switches. |
| **GET** | `/api/v1/matches/{id}/players/{player_id}/features` | Query live Domain 3 multi-cue visual, geometric, and landmark features. |
| **GET** | `/api/v1/teams/{id}/players` | Retrieve players listed under a specific team. |
| **GET** | `/api/v1/players/{id}/analytics` | Retrieve comprehensive speed, distance, and zone metrics for a player. |
| **GET** | `/api/v1/players/{player_id}/features` | Query player feature descriptors directly by player ID. |
| **GET** | `/api/v1/matches/{id}/export/{format}` | Export match summary report as `pdf`, `csv`, or `json`. |

---

## 📓 Fine-Tuning YOLOv11 (Google Colab Notebook)

A standalone Google Colab GPU training notebook is provided in [`notebooks/football_yolo11_training.ipynb`](notebooks/football_yolo11_training.ipynb).
- **Architecture**: Ultralytics YOLOv11 (`yolo11n.pt` / `yolo11s.pt`)
- **Classes**: `player`, `referee`, `ball`, `goalkeeper`
- **Training Strategy**: AdamW optimizer, Mosaic augmentation (0.8), HSV color jitter, and small ball anchor tuning
- **Validation**: Per-class mAP@50 and mAP@50-95 metrics
- **Deployment**: Export to `backend/weights/yolo11_football_finetuned.pt` and set `YOLO_MODEL_PATH=weights/yolo11_football_finetuned.pt` in `.env`.

---

## 🧪 Automated Testing

Run the full backend test suite covering object detection, multi-cue tracking, ReID recovery, and the end-to-end CV pipeline:

```bash
# Run all unit and integration tests
pytest backend/tests -v
```

---

## 🗄️ Database Design

The PostgreSQL relational schema is fully normalized and supports multiple levels of data mapping:
- **Core Entities:** `users`, `matches`, `videos`, `teams`, `players`
- **CV Tracking & Telemetry:** `tracking_data` (stores raw coordinates, speeds, directions, and team colors per player per frame)
- **Analytical Tables:** `heatmaps`, `trajectories`, `analytics`, `processing_jobs` (shares metrics dynamically across match, team, and player levels using scoping foreign keys).

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) to learn how to open issues, set up testing, and submit pull requests.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more details.
