FROM python:3.11-slim-bookworm

# OpenCV / Streamlit-WebRTC runtime libraries (matches packages.txt used for
# Streamlit Community Cloud, plus libglib2.0-0 which opencv-python-headless
# also needs on a minimal base image).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip's own toolchain first -- the base image ships an old
# setuptools that vendors a vulnerable jaraco.context/wheel (CVE-2026-23949,
# CVE-2026-24049), which a container security scan flags regardless of what
# requirements.txt pins. Also drop the stdlib's ensurepip bundled wheels
# (Lib/ensurepip/_bundled/*.whl): they carry their own old pinned
# setuptools (CVE-2025-47273) but only exist to bootstrap pip in a fresh
# venv, which this image never does.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && rm -rf /usr/local/lib/python3.11/ensurepip/_bundled

# Install Python dependencies first so this layer is cached across code changes.
# torch/torchvision are installed explicitly (CPU build) since requirements.txt
# leaves them unpinned on purpose for local dev flexibility (see README.md).
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# App code and runtime assets (training/, tests/, resource/ docs images are
# intentionally excluded via .dockerignore -- not needed at runtime).
COPY Home.py .
COPY .streamlit/ .streamlit/
COPY pages/ pages/
COPY sample_utils/ sample_utils/
COPY models/ models/

RUN mkdir -p /app/temp

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=8501", "--server.address=0.0.0.0"]
