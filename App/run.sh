#!/bin/bash

echo "Uruchamianie orkiestratora treningu."

# Uruchamia główny skrypt, razem z zależnościami
docker-compose run --rm tbi_app python App/run_all.py

echo "Proces zakończony! Wyłączanie baz danych i czyszczenie środowiska."
docker-compose down