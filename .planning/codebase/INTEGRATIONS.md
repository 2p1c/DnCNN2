# External Integrations

**Analysis Date:** 2026-04-12

## Data Input

**File-Based Data Sources:**
- `.mat` files (MATLAB format) - Experimental ultrasonic signal data
  - Noisy signals: 21x21 grid (`data/noisy.mat`)
  - Clean signals: 41x41 grid (`data/clean.mat`)
  - Loaded via `scipy.io.loadmat()` in `scripts/transformer.py`

**Data Processing:**
- Spatial interpolation from 21x21 to 41x41 grid using `scipy.interpolate`
- Supported methods: `cubic` (default), `linear`

## Output Storage

**Local Filesystem Only:**
- Checkpoints: `results/<timestamp>/checkpoints/`
- Training images: `results/<timestamp>/images/`
- Inference results: `results/<timestamp>/` (`.mat` files)

## External Services

**None detected:**
- No cloud storage (S3, GCS, etc.)
- No external APIs (REST, GraphQL)
- No database connections (PostgreSQL, MongoDB, etc.)
- No message queues (Redis, RabbitMQ, etc.)
- No experiment tracking services (Weights & Biases, MLflow, etc.)

## Authentication

**Not applicable:**
- No external services requiring authentication
- No API keys or tokens in configuration

## Compute Infrastructure

**Local compute only:**
- GPU acceleration via CUDA (NVIDIA) or MPS (Apple Silicon)
- CPU fallback when GPU unavailable

## Dependencies

**File I/O:**
- `scipy.io` - MATLAB `.mat` file reading

**No external package integrations:**
- No huggingface_hub or model zoo
- No external dataset repositories
- No pre-trained weights from external sources

## Environment Configuration

**Environment variables (from `.gitignore` patterns):**
- `.env` - Not present (no external service configuration needed)
- No secret management required

**Python path manipulation:**
- `scripts/train/train.py` adds project root to `sys.path` for imports

## CI/CD

**Not detected:**
- No GitHub Actions workflows in `.github/` (workflows directory may exist but no pipeline configuration observed)
- No external CI/CD services

---

*Integration audit: 2026-04-12*
