FROM python:3.13-slim

# `make` drives the pipeline; slim images don't ship it.
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV MPLBACKEND=Agg \
    DISABLE_PANDERA_IMPORT_WARNING=True \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Runs the full reorganised workflow end-to-end (generates synthetic data if the
# real creditcard.csv is not mounted into data/raw/).
CMD ["make", "pipeline"]
