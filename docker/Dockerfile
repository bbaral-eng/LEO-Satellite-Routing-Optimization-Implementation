
FROM python:3.11-slim-bookworm AS builder

# Install build dependencies, debian has this god awful time check that causes issues, so I had to include the fix for it here based on a bug fix I saw online. 
RUN echo 'Acquire::Max-FutureTime 2592000;' > /etc/apt/apt.conf.d/10fix-future \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        cmake \
        make \
        g++ \
        libeigen3-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything needed to build
COPY requirements.txt pyproject.toml CMakeLists.txt ./
COPY src/ src/

# Install build system + runtime deps, then build & install the C++ extension
RUN pip install --no-cache-dir scikit-build-core pybind11 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-build-isolation .


# runtime image 
FROM python:3.11-slim-bookworm

WORKDIR /app

# Copy installed Python packages (including the compiled core_engine .so)
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project source 
COPY configs/    configs/
COPY examples/   examples/
COPY src/        src/

# Default command, can be overridden in docker-compose.yml
CMD ["python", "examples/run_98_node_sim.py", "--first", "5"]
