# Bean Leaf Disease Classifier

Built around SDG 2 (Zero Hunger) and SDG 1 (No Poverty): a phone photo of a
bean leaf can flag disease early, before a harvest is lost. Beans are a
staple crop and a main protein source for many East African households,
including in Rwanda.

- Video demo: https://youtu.be/LwHj1jq8iAQ
- Live API/UI: https://bean-disease-classifier.onrender.com

## Project description

This project takes an image classifier through training and evaluation, an
API for prediction and retraining, a monitoring UI, Docker deployment, and
load testing.

The model labels bean leaf photos as one of:
- `healthy`
- `angular_leaf_spot`
- `bean_rust`

Dataset: [iBean](https://github.com/AI-Lab-Makerere/ibean) from Makerere AI
Lab / NaCRRI. 1,296 field-collected leaf images across those three classes,
loaded with Hugging Face `datasets` (`AI-Lab-Makerere/beans`).

## Repository structure

```
bean-disease-classifier/
├── README.md
├── requirements.txt
├── Dockerfile
├── entrypoint.sh             # starts uvicorn on $PORT
├── render.yaml               # Render Blueprint (free plan)
├── requirements-docker.txt   # slim API image deps (no Locust/notebook stack)
├── docker-compose.yml
├── nginx.conf                # load balancer for scaled API replicas
├── notebook/
│   └── bean_disease_classifier.ipynb   # training + evaluation pipeline
├── src/
│   ├── preprocessing.py      # data acquisition + preprocessing (shared w/ API)
│   ├── model.py               # model build / train / retrain
│   └── prediction.py          # single-image inference
├── data/
│   ├── train/                  # per-class training images
│   └── test/                   # per-class test images
├── models/
│   └── bean_model_latest.h5   # current model (+ versioned copies)
├── api/
│   ├── main.py                 # FastAPI (predict / upload / retrain / status)
│   └── database.py             # SQLite logging
├── ui/
│   ├── index.html               # monitoring + prediction + retraining dashboard
│   └── img/                     # exploration plots saved by the notebook (Data tab)
├── sample_images/
│   ├── predict/                 # one sample leaf per class for manual demos
│   └── retrain/                 # a few extras for upload/retrain demos
└── locust/
    ├── locustfile.py            # load test script
    └── results/                 # Locust CSV results per container count
```

## Setup

> **Note:** `data/train` and `data/test` are already in the repo.
> `data/train` holds the iBean train and validation splits, and `data/test`
> holds the iBean test split. The API can predict and retrain straight away;
> you only need the notebook to retrain the model from scratch.

### 1. Notebook (train the model)

The notebook downloads the dataset from Hugging Face, so it needs normal
internet access. Prefer
[Google Colab](https://colab.research.google.com/)
(Runtime → Change runtime type → GPU recommended, not required).

The first code cell sets up Colab. If it detects Colab, it clones this repo
into `/content/Bean_disease_classifier` and changes into its `notebook/`
folder, so the `../src` imports and `../data` paths work the same as they do
locally. When run locally, the cell does nothing.

```bash
# If running locally instead of Colab:
pip install -r requirements.txt
jupyter notebook notebook/bean_disease_classifier.ipynb
```

Run all cells top to bottom. That will:
1. Download iBean and export it to `data/train`, `data/test`
2. Show class distribution, sample images, and brightness plots
3. Train a MobileNetV2 transfer-learning model
4. Evaluate on the test set (accuracy, precision, recall, F1, confusion
   matrix, ROC-AUC)
5. Save the model to `models/bean_model_latest.h5`

### 2. API + UI (local, no Docker)

```bash
pip install -r requirements.txt
# put your trained bean_model_latest.h5 in models/ first
cd api
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000` for the dashboard (Status / Data / Predict /
Upload & retrain). Swagger docs: `http://localhost:8000/docs`.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| GET | `/status` | Uptime, model version, prediction/upload stats, retrain history. Also `uploads_since_last_retrain` and `retrain_recommended` (true once that count reaches `RETRAIN_THRESHOLD`, default 10) |
| POST | `/predict` | Upload one image (`file`) → class + confidence |
| POST | `/upload-data/{class_name}` | Bulk-upload images (`files`) for a class. `class_name` must be `angular_leaf_spot`, `bean_rust`, or `healthy`. Files that are not valid images are skipped and listed in `rejected` |
| POST | `/retrain?epochs=5` | Retrain on current `data/train` (original + uploaded images). The old and new models are scored on a fixed `data/test` subset; returns `old_test_accuracy`, `new_test_accuracy`, and `promoted` (the new model is kept only if it scores at least as well) |
| GET | `/uploads` | The 50 most recent uploads (class, filename, time) |
| GET | `/visualizations/class-distribution` | Image counts per class |

Uploads and predictions are logged to SQLite (`api/app_data.db`) for the
dashboard.

### 3. Docker (single container)

```bash
docker build -t bean-classifier .
# Linux / macOS:
docker run -p 8000:8000 -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data bean-classifier
# Windows PowerShell:
# docker run -p 8000:8000 -v ${PWD}/models:/app/models -v ${PWD}/data:/app/data bean-classifier
```

Then open `http://localhost:8000`.

### 3b. Deploy on Render

One Docker web service (API + UI). Render does not run
`docker-compose --scale`; paid plans use "multiple instances" instead. For
this assignment, one instance is enough.

`render.yaml` uses the Free plan (512 MB). The free tier sleeps when idle,
so the first load after a quiet spell takes about a minute while it wakes
up. Heavy load or retraining can also run out of memory. That is fine for a
demo URL.

1. Push this repo to GitHub (include `models/bean_model_latest.h5` and `data/`).
2. In the [Render Dashboard](https://dashboard.render.com) → **New** →
   **Blueprint** → connect the repo (uses `render.yaml`), or New → Web
   Service → Docker.
3. Confirm instance type is Free, health check `/status`.
4. Deploy. The first build is slow because of TensorFlow.
5. Open https://bean-disease-classifier.onrender.com.
6. Keep that URL on the Live API/UI line at the top of this README.

`entrypoint.sh` binds uvicorn to Render's `$PORT`. On Free, uploads and
retrains reset whenever the instance restarts or redeploys.

Local smoke test of the Render-style start:

```bash
docker build -t bean-classifier .
docker run -p 8000:8000 -e PORT=8000 bean-classifier
```

### 4. Docker Compose (scalable, load-balanced)

`docker-compose.yml` puts the API behind nginx so you can scale replicas and
spread traffic (round-robin via Docker DNS; see `nginx.conf`).

```bash
docker compose up --build --scale api=1   # 1 replica
# or
docker compose up --build --scale api=4   # 4 replicas
```

The app is at `http://localhost:8080` through nginx, regardless of replica
count.

### 5. Load testing (Locust)

With the stack running (plain Docker or compose):

```bash
cd locust
pip install locust
locust -f locustfile.py --host=http://localhost:8080 \
    --headless -u 50 -r 10 -t 60s \
    --csv=results/run_4containers
```

Re-run once per replica count (`--scale api=1`, `2`, `4`), change the
`--csv` prefix each time, then compare the `*_stats.csv` files. On a machine
with enough CPU and RAM, median and p95 latency usually fall and requests/sec
rise as replicas increase, until the host itself is the bottleneck.

## Results

Notebook evaluation on the held-out iBean test set (128 images):

| Metric | Value |
|---|---|
| Accuracy | 89.8% |
| Macro precision | 0.914 |
| Macro recall | 0.899 |
| Macro F1 | 0.898 |

Per-class breakdown:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Angular Leaf Spot | 0.796 | 1.000 | 0.887 |
| Bean Rust | 0.970 | 0.744 | 0.842 |
| Healthy | 0.976 | 0.952 | 0.964 |

The model never misses a real Angular Leaf Spot case (perfect recall) but
sometimes over-predicts it, which hurts Bean Rust recall. Most mistakes sit
between those two look-alike diseases; Healthy is classified reliably. For a
disease alert tool that tradeoff is acceptable: a false alarm beats a missed
outbreak.

Confusion matrix, ROC curves, and training curves are in the notebook
(Section 5).

## Flood request simulation results

Locust headless against the nginx entrypoint (`http://localhost:8080`),
hitting `/predict`, `/status`, and `/visualizations/class-distribution`.
Scale 1-2: 6 users, spawn rate 2/s, 45s. Scale 4: 4 users, 60s. Host:
Windows laptop about 4 GB RAM, Docker Desktop memory limit about 1.8 GB.

| API replicas | Total requests | Failures | Req/s (agg.) | Median `/predict` | p95 `/predict` |
|---|---|---|---|---|---|
| 1 | 55 | 0 (0%) | 1.34 | 1,900 ms | 12,000 ms |
| 2 | 26 | 0 (0%) | 0.67 | 2,200 ms | 26,000 ms |
| 4 | 4 | 4 (100%) | 0.08 | timeouts (~45 s) | n/a (all timed out) |

Raw CSVs: `locust/results/run_1containers_stats.csv`,
`run_2containers_stats.csv`, `run_4containers_stats.csv`.

On this machine, one replica performed best. A second replica did not raise
throughput: TF inference already saturated the shared CPU, so nginx spreading
load mostly added queueing (higher average and p95 latency). At four
replicas the Docker VM was oversubscribed (about 250-500 MB RSS per API
process inside a 1.8 GB limit), and almost every request timed out. On a
larger host, the same `--scale api=N` setup should show rising req/s and
falling p95 as N grows, because nginx round-robins across healthy replicas.
