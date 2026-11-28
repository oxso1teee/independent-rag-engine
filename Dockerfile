# syntax=docker/dockerfile:1
# Multi-stage image of the HTTP API (`rag serve`). The base image is pinned by digest, the
# dependencies by hash (from uv.lock), and the final stage runs as an unprivileged user.
#
#   docker build -t rag-engine .
#   docker run --rm -p 127.0.0.1:8765:8765 -e RAG_API_TOKEN="$(openssl rand -hex 24)" \
#       -v rag-data:/data rag-engine
#
# Extras go in at build time: --build-arg EXTRAS="--extra api --extra onnx" adds the ONNX
# embedding runtime (about 250 MB more; the default image reaches Ollama over HTTP instead).

ARG PYTHON_IMAGE=python:3.12-slim@sha256:191083cdf6c7bda8c76158983163ead5d5cf7b77cf3033e805bb0f5296f51b4b

FROM ${PYTHON_IMAGE} AS build
ARG UV_VERSION=0.11.18
ARG EXTRAS="--extra api"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1 UV_NO_PROGRESS=1
RUN pip install "uv==${UV_VERSION}"
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv export --frozen --no-dev --no-emit-project ${EXTRAS} --format requirements-txt \
        -o /tmp/requirements.txt \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --require-hashes --no-deps -r /tmp/requirements.txt \
    && uv build --wheel --out-dir /tmp/wheel \
    && /opt/venv/bin/pip install --no-deps /tmp/wheel/*.whl

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --system --uid 10001 --home-dir /data --shell /usr/sbin/nologin rag \
    && mkdir /data && chown rag:rag /data
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_STORAGE__DB_PATH=/data/rag.db \
    RAG_API__HOST=0.0.0.0
USER rag
WORKDIR /data
VOLUME ["/data"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/v1/health', timeout=4).status == 200 else 1)"]
# A non-loopback bind refuses to start without RAG_API_TOKEN (at least 16 characters).
ENTRYPOINT ["rag"]
CMD ["serve"]
