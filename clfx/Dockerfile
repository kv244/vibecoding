# ─── Stage 1: Build the C engine ─────────────────────────────────────────────
FROM ubuntu:22.04 AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    opencl-headers \
    ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY main.c .
COPY include/ include/

# Compile to a standalone Linux binary (no Cygwin, no Windows deps)
RUN gcc -Iinclude main.c -o clfx -lOpenCL -lm -std=c99 -D_POSIX_C_SOURCE=200112L

# ─── Stage 2: Runtime image ───────────────────────────────────────────────────
FROM ubuntu:22.04

# pocl = portable OpenCL -- runs on any CPU, no GPU required
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    pocl-opencl-icd \
    ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy compiled engine from builder stage
COPY --from=builder /build/clfx ./clfx
RUN chmod +x clfx

# Copy GUI source
COPY gui/ gui/
COPY visualize.py .

# Install Python dependencies
RUN pip3 install --no-cache-dir flask gunicorn

# Cloud Run injects PORT env var (default 8080)
ENV PORT=8080

# Two workers: one for audio processing (slow), one for GUI requests
CMD exec gunicorn \
      --bind "0.0.0.0:${PORT}" \
      --timeout 300 \
      --workers 2 \
      --log-level info \
      "gui.server:app"
