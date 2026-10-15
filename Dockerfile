FROM python:3.12.12-slim-bookworm AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build
COPY requirements.txt constraints.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venv/bin/pip install -r requirements.txt \
    && /opt/venv/bin/pip check

FROM python:3.12.12-slim-bookworm AS runtime
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /service
RUN groupadd --gid 10001 shrimp \
    && useradd --uid 10001 --gid shrimp --create-home shrimp \
    && mkdir -p /service/models \
    && chown shrimp:shrimp /service/models
COPY --from=builder /opt/venv /opt/venv
COPY --chown=shrimp:shrimp app ./app
COPY --chown=shrimp:shrimp training ./training
COPY --chown=shrimp:shrimp scripts ./scripts
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/ready', timeout=4)"
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
