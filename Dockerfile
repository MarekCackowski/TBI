FROM docker.io/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Instalacja niezbędnych narzędzi systemowych (dodano aria2 do pobierania wielowątkowego)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# Wyłączenie buforowania wyjścia Pythona dla wszystkich skryptów w kontenerze
ENV PYTHONUNBUFFERED=1

# Ustawienie katalogu roboczego w kontenerze
WORKDIR /app

# Aktualizacja narzędzi budujących
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalacja MONAI i Einops z pominięciem sprawdzania zależności (aby nie pobierać PyTorcha)
RUN pip install --no-cache-dir --no-deps monai>=1.3.0 einops>=0.7.0

# Kopiowanie pliku z wymaganiami i instalacja reszty lekkich bibliotek
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=2000 -r requirements.txt

# Kopiowanie kodu projektu do kontenera
COPY . .

# Uruchomienie całości treningu
CMD ["python3", "-B", "App/RunAll.py"]