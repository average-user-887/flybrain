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
COPY validation/ ./validation/
COPY neurofly_studio/ ./neurofly_studio/
COPY data-provenance/ ./data-provenance/
COPY docs/CAPABILITY_MATRIX.md ./docs/CAPABILITY_MATRIX.md
COPY web/ ./web/
COPY *.py ./

# Install neurofly with embodied physics and testing dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[body,test]"

# Default port for neurofly daemon
EXPOSE 8769

# Data. The image ships no MaleCNS data. Without a verified graph `neurofly run` starts the
# hand-built modular controller and says so. For the connectome, keep the data on the host
# and mount it (the default locations inside the image are under /app):
#   docker run --rm -v "$PWD/connectome_data:/app/connectome_data" neurofly:latest download-data
#   docker run --rm -v "$PWD/connectome_data:/app/connectome_data" -v "$PWD/outputs:/app/outputs" \
#       --entrypoint python neurofly:latest -m brainlab.connectome
#   docker run --rm -v "$PWD/connectome_data:/app/connectome_data" -v "$PWD/outputs:/app/outputs" \
#       --entrypoint python neurofly:latest -m brainlab.prepare
#   docker run -p 127.0.0.1:8769:8769 -v "$PWD/connectome_data:/app/connectome_data" \
#       -v "$PWD/outputs:/app/outputs" neurofly:latest

# Default entrypoint
ENTRYPOINT ["neurofly"]
# Inside the container the daemon must listen on all interfaces for `-p` to reach it;
# publish with `-p 127.0.0.1:8769:8769` to keep it off the LAN.
CMD ["run", "--host", "0.0.0.0", "--port", "8769", "--paradigm", "multisensory-sandbox"]
