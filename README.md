## How to Use

### Docker-way

```shell
cp docker-compose.dist.yml docker-compose.yml
docker-compose up -d
```

```shell
docker-compos exec api bash
```

Download model:

```shell
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir models/fish-speech-1.5/
```

### Manual

```shell
apt install cmake portaudio19-dev
```

Prepare virtual environment

Install all dependencies:

```shell
pip install -r requirements.txt
```

Download model:

```shell
huggingface-cli download fishaudio/fish-speech-1.5 --local-dir models/fish-speech-1.5/
```

## Testing

Using default model:

```shell
curl http://gpu02:8000/v1/audio/speech \
  -X POST \
  -F model="fish-speech-1.5" \
  -F input="Hello, this is a test of Fish Speech API" \
  --output "speech.wav"
```

Using reference sample:

```shell
curl http://localhost:8000/v1/audio/speech \
  -X POST \
  -F model="fish-speech-1.5" \
  -F input="Hello, this is a test of Fish Speech API" \
  -F instructions="cheerful" \
  -F reference_audio="@reference.wav" \
  --output "speech.wav"
```
