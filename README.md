# 🐟 Fish Speech API

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
git clone git@github.com:EvilFreelancer/fish-speech-api.git
cd fish-speech-api
```

### Docker-way

```shell
cp docker-compose.dist.yml docker-compose.yml
docker-compose up -d
```

Enter the container:

```shell
docker-compos exec api bash
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

Generate speech with default voice:
  
```shell
curl http://gpu02:8000/v1/audio/speech \
  -X POST \
  -F model="fish-speech-1.5" \
  -F input="Hello, this is a test of Fish Speech API" \
  --output "speech.wav"
```

Generate speech with voice style transfer:

```shell
curl http://gpu02:8000/v1/audio/speech \
  -X POST \
  -H 'Content-Type: multipart/form-data' \
  -F model="fish-speech-1.5" \
  -F input="Никита, это уже вариант с подменой голоса. Работает чуть дольше по времени, чем оригинальная версия, но зато работает." \
  -F reference_audio="@./audio_2025-04-12_21-01-04.wav" \
  --output "speech.wav"
```

Advanced settings:

```shell
curl http://gpu02:8000/v1/audio/speech \
  -X POST \
  -H 'Content-Type: multipart/form-data' \
  -F model="fish-speech-1.5" \
  -F input="Привет! Это тест голоса." \
  -F top_p="0.8" \
  -F repetition_penalty="1.3" \
  -F temperature="0.75" \
  -F chunk_length="150" \
  -F max_new_tokens="768" \
  -F seed="42" \
  -F reference_audio="@./audio_2025-04-12_21-01-04.wav" \
  --output "speech.wav"
```

## Links

- https://github.com/fishaudio/fish-speech
- https://huggingface.co/fishaudio/fish-speech-1.5
- https://huggingface.co/fishaudio/fish-agent-v0.1-3b
