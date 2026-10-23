# Wybór bazowego obrazu z PyTorch i CUDA
FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Wyłączenie buforowania wyjścia Pythona dla wszystkich skryptów w kontenerze
ENV PYTHONUNBUFFERED=1

# Ustawienie katalogu roboczego w kontenerze
WORKDIR /app

# Instalacja narzędzi systemowych
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# Aktualizacja narzędzi Pythona
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalacja kluczowych bibliotek głębokiego uczenia
RUN pip install --no-cache-dir --no-deps monai>=1.3.0 einops>=0.7.0

# Najpierw kopiujemy plik z wymaganiami, a nie cały kod
COPY requirements.txt .

# Instalacja zależności z requirements.txt
RUN pip install --no-cache-dir --default-timeout=2000 -r requirements.txt


COPY . .

# Uruchomienie całości treningu
CMD ["python3", "-B", "App/RunAll.py"]