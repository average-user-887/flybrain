# Project NeuroFly — Reproducible Embodied Co-Simulation Environment
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    NEUROFLY_RUN_PHYSICS=1

# Install system dependencies for MuJoCo, FlyGym, and OpenGL headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0 \
    libegl1 \
    libgles2 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml README.md NOTICE LICENSE ./
COPY neurofly/ ./neurofly/
COPY brainlab/ ./brainlab/
COPY experiments/ ./experiments/
COPY neurofly_body/ ./neurofly_body/
COPY web/ ./web/
COPY *.py ./

# Install neurofly with embodied physics and testing dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[body,test]"

# Default port for neurofly daemon
EXPOSE 8769

# Default entrypoint
ENTRYPOINT ["neurofly"]
CMD ["run", "--port", "8769", "--paradigm", "multisensory-sandbox"]
