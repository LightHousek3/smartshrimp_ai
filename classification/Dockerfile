FROM python:3.12.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 RUNTIME_DIR=/tmp/runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 shrimp && useradd --uid 10001 --gid shrimp --no-create-home shrimp
WORKDIR /service
COPY classification/requirements.txt classification/constraints.txt /service/classification/
RUN python -m pip install -c classification/constraints.txt -r classification/requirements.txt && python -m pip check
COPY classification /service/classification
COPY LICENSE NOTICE /service/
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/ready', timeout=4)" || exit 1
CMD ["python", "-m", "uvicorn", "classification.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
