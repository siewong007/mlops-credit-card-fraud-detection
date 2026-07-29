FROM python:3.13.9-slim@sha256:326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86

# `make` drives the verification gate; slim images do not include it.
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG SOURCE_COMMIT
ENV SOURCE_COMMIT=${SOURCE_COMMIT} \
    MPLBACKEND=Agg \
    DISABLE_PANDERA_IMPORT_WARNING=True \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["make", "verify"]
