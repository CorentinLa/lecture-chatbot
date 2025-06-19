FROM python:3.11-slim

# Installer build-essential si nécessaire (par ex. pour chromadb ou autres dépendances C++)
RUN apt-get update && apt-get install -y build-essential

# Définir le dossier de travail
WORKDIR /app

# Copier les fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code de l'app
COPY . .

# Exposer le port pour FastAPI
EXPOSE 8000

# Commande pour démarrer le serveur FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]