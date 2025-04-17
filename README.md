# Fish Speech API Webserver in Docker

OpenAPI-like voice generation server based on [fish-speech-1.5](https://huggingface.co/fishaudio/fish-speech-1.5).

Supports `text-to-speech` and voice style transfer via reference audio samples.

## Requirements

* Nvidia GPU
* For Docker-way
    * Nvidia Docker Runtime
    * Docker
    * Docker Compose
* For Manual Setup
    * Python 3.12
    * Python Venv

## 🔧 Quick Start

Clone the repo first:

```shell
git clone --recurse-submodules git@github.com:EvilFreelancer/fish-speech-api.git
cd docker-fish-speech-server
```

### Docker-way

```shell
cp docker-compose.dist.yml docker-compose.yml
docker compose up -d
```

Enter the container:

```shell
docker compose exec api bash
```

Download the model:

```shell
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir models/fish-speech-1.5/
```

### Manual Setup

```shell
apt install cmake portaudio19-dev
```

Set up a virtual environment and install dependencies:

```shell
python3.12 -m venv venv
pip install -r requirements.txt
```

Download model:

```shell
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir models/fish-speech-1.5/
```

Run API-server:

```shell
python main.py
```

## 🧪 Testing the API

### Generate speech with default voice

```shell
curl http://localhost:8000/audio/speech \
  -X POST \
  -F model="fish-speech-1.5" \
  -F input="Hello, this is a test of Fish Speech API" \
  --output "speech.wav"
```

In JSON format:

```shell
curl http://localhost:8000/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
      "model": "fish-speech-1.5",
      "input": "Hello, this is a test of Fish Speech API"
  }' \
  --output "speech.wav"
```

### Generate speech with voice style transfer

```shell
curl http://localhost:8000/audio/speech \
  -X POST \
  -H 'Content-Type: multipart/form-data' \
  -F model="fish-speech-1.5" \
  -F input="Hello, this is a test of Fish Speech API" \
  -F reference_audio="@voice-viola.wav" \
  --output "speech.wav"
```

In JSON format:

```shell
curl http://localhost:8000/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
      "model": "fish-speech-1.5",
      "input": "Hello, this is a test of Fish Speech API",
      "reference_audio": "=base64..."
  }' \
  --output "speech.wav"
```

#### Advanced settings

```shell
curl http://localhost:8000/audio/speech \
  -X POST \
  -H 'Content-Type: multipart/form-data' \
  -F model="fish-speech-1.5" \
  -F input="Hello, this is a test of Fish Speech API" \
  -F top_p="0.1" \
  -F repetition_penalty="1.3" \
  -F temperature="0.75" \
  -F chunk_length="150" \
  -F max_new_tokens="768" \
  -F seed="42" \
  -F reference_audio="@voice-viola.wav" \
  --output "speech.wav"
```

In JSON format:

```shell
curl http://localhost:8000/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
      "model": "fish-speech-1.5",
      "input": "Hello, this is a test of Fish Speech API",
      "top_p": "0.1",
      "repetition_penalty": "1.3",
      "temperature": "0.75",
      "chunk_length": "150",
      "max_new_tokens": "768",
      "seed": "42",
      "reference_audio": "=base64..."
  }' \
  --output "speech.wav"
```

## Links

- https://github.com/fishaudio/fish-speech
- https://huggingface.co/fishaudio/fish-speech-1.5
- https://huggingface.co/fishaudio/fish-agent-v0.1-3b
