"""
FastAPI service for the Bean Leaf Disease Classifier.

Endpoints:
  GET  /status
  POST /predict
  POST /upload-data/{class_name}
  POST /retrain
  GET  /uploads
  GET  /visualizations/class-distribution

Run locally:
    uvicorn main:app --reload --port 8000
"""

import gc
import os
import shutil
import sys
import tempfile
import time

# Keep TensorFlow lean before it is imported by src modules.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import (  # noqa: E402
    CLASS_NAMES,
    load_dataset_from_directory,
    build_pipeline,
    prepare_retrain_subset,
    preprocess_image,
)
from model import retrain_model  # noqa: E402
from prediction import predict_image, clear_model_cache  # noqa: E402

import database  # noqa: E402

APP_START_TIME = time.time()

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_DEFAULT_TRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "train")
MODEL_DIR = os.environ.get("BEAN_MODEL_DIR", _DEFAULT_MODEL_DIR)
DATA_TRAIN_DIR = os.environ.get("BEAN_TRAIN_DIR", _DEFAULT_TRAIN_DIR)
DATA_TEST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test")
MODEL_PATH = os.path.join(MODEL_DIR, "bean_model_latest.h5")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# Free Render is ~512 MB. Full-set retrain OOMs; keep a small subset.
RETRAIN_MAX_PER_CLASS = int(os.environ.get("RETRAIN_MAX_PER_CLASS", "12"))
RETRAIN_BATCH_SIZE = int(os.environ.get("RETRAIN_BATCH_SIZE", "2"))
RETRAIN_MAX_EPOCHS = int(os.environ.get("RETRAIN_MAX_EPOCHS", "2"))
RETRAIN_THRESHOLD = int(os.environ.get("RETRAIN_THRESHOLD", "10"))

# Fixed test subset used to compare the old and new model on /retrain.
EVAL_IMAGES_PER_CLASS = 10
EVAL_BATCH_SIZE = 8


def _count_images(directory: str) -> int:
    """Count image files in a directory, ignoring placeholders like .gitkeep."""
    if not os.path.isdir(directory):
        return 0
    return sum(
        1
        for name in os.listdir(directory)
        if os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS
    )


def _load_eval_subset():
    """First EVAL_IMAGES_PER_CLASS test images per class, sorted by filename."""
    paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_TEST_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        names = sorted(
            n for n in os.listdir(class_dir)
            if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS
        )
        for name in names[:EVAL_IMAGES_PER_CLASS]:
            paths.append(os.path.join(class_dir, name))
            labels.append(label)

    def load(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        return preprocess_image(image, label)

    ds = tf.data.Dataset.from_tensor_slices((paths, tf.constant(labels, tf.int64)))
    return ds.map(load).batch(EVAL_BATCH_SIZE)


def _test_accuracy(model_path: str) -> float:
    """Load a saved model, score it on the fixed test subset, then free it."""
    model = tf.keras.models.load_model(model_path)
    _, accuracy = model.evaluate(_load_eval_subset(), verbose=0)
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    return float(accuracy)


app = FastAPI(
    title="Bean Leaf Disease Classifier API",
    description="Predict bean leaf disease from an image, and manage retraining.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # relaxed for the demo UI; tighten for real production use
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    database.init_db()
    os.makedirs(MODEL_DIR, exist_ok=True)
    for class_name in CLASS_NAMES:
        os.makedirs(os.path.join(DATA_TRAIN_DIR, class_name), exist_ok=True)

    try:
        import tensorflow as tf

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except Exception:
        pass


UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")


@app.get("/")
def root():
    index_path = os.path.join(UI_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"service": "Bean Leaf Disease Classifier API", "docs": "/docs"}


@app.get("/status")
def status():
    """
    Model uptime, current model version, and prediction/upload stats -
    powers the UI's monitoring dashboard.
    """
    uptime_seconds = time.time() - APP_START_TIME

    model_version = "no model found"
    if os.path.exists(MODEL_PATH):
        model_version = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(MODEL_PATH))
        )

    pred_stats = database.get_prediction_stats()
    upload_dist = database.get_upload_class_distribution()
    retrain_history = database.get_retrain_history()
    uploads_since_retrain = database.count_uploads_since_last_retrain()

    return {
        "status": "up",
        "uptime_seconds": round(uptime_seconds, 1),
        "model_last_updated": model_version,
        "total_predictions_served": pred_stats["total"],
        "predictions_by_class": pred_stats["by_class"],
        "recent_predictions": pred_stats["recent"],
        "uploaded_images_by_class": upload_dist,
        "retrain_history": retrain_history,
        "uploads_since_last_retrain": uploads_since_retrain,
        "retrain_recommended": uploads_since_retrain >= RETRAIN_THRESHOLD,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predicts the disease class of a single uploaded bean leaf image.
    """
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    image_bytes = await file.read()
    try:
        result = predict_image(image_bytes=image_bytes, model_path=MODEL_PATH)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")

    database.log_prediction(result["predicted_class"], result["confidence"])
    return result


@app.post("/upload-data/{class_name}")
async def upload_data(class_name: str, files: list[UploadFile] = File(...)):
    """
    Bulk-uploads images for a given class, to be used in the next
    retraining run. Saves files to data/train/<class_name>/ and logs
    each upload to the database. Files that are not valid images are
    skipped and returned in "rejected".
    """
    if class_name not in CLASS_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid class_name. Must be one of {CLASS_NAMES}",
        )

    class_dir = os.path.join(DATA_TRAIN_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

    saved = []
    rejected = []
    for f in files:
        # basename() drops any folder parts like "../" from the client's name.
        original_name = os.path.basename(f.filename or "")
        image_bytes = await f.read()
        try:
            tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        except Exception:
            rejected.append(original_name)
            continue

        timestamp = int(time.time() * 1000)
        safe_name = f"uploaded_{timestamp}_{original_name}"
        dest_path = os.path.join(class_dir, safe_name)
        with open(dest_path, "wb") as out_file:
            out_file.write(image_bytes)
        database.log_upload(class_name, safe_name)
        saved.append(safe_name)

    return {
        "class_name": class_name,
        "num_uploaded": len(saved),
        "filenames": saved,
        "rejected": rejected,
    }


@app.post("/retrain")
def retrain(epochs: int = 1):
    """
    Triggers lightweight retraining for small cloud hosts.

    Uses a capped per-class subset (uploads preferred) so Free-tier
    Render (512 MB) does not OOM on the full training folder.

    The old and new models are both scored on the same fixed test subset,
    and the new model is only kept if it is at least as accurate.
    """
    total_images = sum(
        _count_images(os.path.join(DATA_TRAIN_DIR, c)) for c in CLASS_NAMES
    )
    if total_images < 20:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Not enough training data to retrain "
                f"({total_images} images found, need >= 20)."
            ),
        )

    epochs = max(1, min(int(epochs), RETRAIN_MAX_EPOCHS))
    work_dir = tempfile.mkdtemp(prefix="bean_retrain_")

    try:
        # Free RAM held by the cached predict model before loading for train.
        clear_model_cache()
        gc.collect()

        selected = prepare_retrain_subset(
            DATA_TRAIN_DIR, work_dir, max_per_class=RETRAIN_MAX_PER_CLASS
        )
        subset_total = sum(selected.values())
        if subset_total < 12:
            raise HTTPException(
                status_code=400,
                detail=f"Retrain subset too small ({subset_total} images).",
            )

        # Score the current model and keep a backup in case the new one is worse.
        # (done after prepare_retrain_subset, which clears work_dir)
        old_test_accuracy = None
        backup_path = os.path.join(work_dir, "previous_model.h5")
        if os.path.exists(MODEL_PATH):
            old_test_accuracy = _test_accuracy(MODEL_PATH)
            shutil.copy2(MODEL_PATH, backup_path)

        train_raw, val_raw = load_dataset_from_directory(work_dir)
        train_ds = build_pipeline(
            train_raw,
            batch_size=RETRAIN_BATCH_SIZE,
            augment=False,
            shuffle=True,
            shuffle_buffer=32,
        )
        val_ds = build_pipeline(
            val_raw,
            batch_size=RETRAIN_BATCH_SIZE,
            augment=False,
            shuffle=False,
            shuffle_buffer=32,
        )

        history, versioned_path, latest_path = retrain_model(
            train_ds,
            val_ds,
            model_dir=MODEL_DIR,
            epochs=epochs,
            lightweight=True,
        )

        new_test_accuracy = _test_accuracy(latest_path)
        promoted = old_test_accuracy is None or new_test_accuracy >= old_test_accuracy
        if not promoted:
            shutil.copy2(backup_path, MODEL_PATH)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Retraining failed: {e}. "
                "On Render Free (512 MB) use 1 epoch and upload only a few images."
            ),
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        clear_model_cache()
        gc.collect()

    final_val_accuracy = None
    val_accs = history.history.get("val_accuracy") or []
    if val_accs:
        final_val_accuracy = float(val_accs[-1])

    database.log_retrain_event(
        num_new_images=subset_total,
        epochs=epochs,
        final_val_accuracy=final_val_accuracy,
        model_version=os.path.basename(versioned_path),
        old_test_accuracy=old_test_accuracy,
        new_test_accuracy=new_test_accuracy,
        promoted=promoted,
    )

    return {
        "status": "retrained",
        "model_version": os.path.basename(versioned_path),
        "final_val_accuracy": final_val_accuracy,
        "old_test_accuracy": old_test_accuracy,
        "new_test_accuracy": new_test_accuracy,
        "promoted": promoted,
        "trained_on_images": subset_total,
        "subset_by_class": selected,
        "note": (
            f"Used a memory-safe subset (max {RETRAIN_MAX_PER_CLASS}/class). "
            "Uploaded images are preferred."
        ),
    }


@app.get("/uploads")
def recent_uploads():
    """The 50 most recent uploads (class, filename, timestamp), newest first."""
    return database.get_recent_uploads(limit=50)


@app.get("/visualizations/class-distribution")
def class_distribution():
    """
    Current class counts in data/train (original training data + any
    images uploaded via /upload-data). Used by the UI monitoring charts.
    """
    counts = {}
    for class_name in CLASS_NAMES:
        counts[class_name] = _count_images(os.path.join(DATA_TRAIN_DIR, class_name))
    return counts
