# Wybór bazowego obrazu z PyTorch i CUDA
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Wyłączenie buforowania wyjścia Pythona
ENV PYTHONUNBUFFERED=1

# Ustawienie katalogu roboczego
WORKDIR /app

# Instalacja narzędzi systemowych (w tym aria2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# Aktualizacja menedżera pakietów i instalacja MONAI / Einops
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --no-deps monai>=1.3.0 einops>=0.7.0

# Kopiowanie i instalacja reszty zależności
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=2000 -r requirements.txt

# Kopiowanie całego kodu projektu do kontenera
COPY . .

# Uruchomienie orkiestratora
CMD ["python3", "-B", "App/RunAll.py"]