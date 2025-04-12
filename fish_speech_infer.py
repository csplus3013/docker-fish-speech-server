import os
import sys
import time
import torch
import soundfile as sf
import torchaudio
import tempfile
import shutil
import gc
from contextlib import contextmanager
from huggingface_hub import snapshot_download

FISH_SPEECH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fish_speech")
sys.path.append(FISH_SPEECH_DIR)

from fish_speech.tokenizer import FishTokenizer
from fish_speech.models.text2semantic.inference import main as text2semantic_main
from fish_speech.models.vqgan import inference as vqgan_inference


@contextmanager
def temporary_argv(new_argv):
    original_argv = sys.argv
    sys.argv = new_argv
    try:
        yield
    finally:
        sys.argv = original_argv


def download_models(cache_dir="./models/fish-speech-1.5", local_only=True):
    try:
        repo_dir = snapshot_download(
            repo_id="fishaudio/fish-speech-1.5",
            cache_dir=cache_dir,
            local_files_only=local_only
        )
        return repo_dir
    except Exception as e:
        print(f"Model download failed: {e}")
        return None


def encode_reference_audio(reference_audio_path, temp_dir, device="cuda"):
    waveform, sample_rate = torchaudio.load(reference_audio_path)
    waveform = waveform.to(device)

    if waveform.abs().max() > 1.0:
        waveform = waveform / waveform.abs().max()

    audio_int16 = (waveform * 32767).to(torch.int16).cpu().numpy()
    normalized_path = os.path.join(temp_dir, "normalized_ref.wav")
    sf.write(normalized_path, audio_int16.T, sample_rate)

    reference_tokens_path = os.path.join(temp_dir, "reference_tokens.npy")
    vqgan_inference.encode_audio(normalized_path, reference_tokens_path)

    return reference_tokens_path


def generate_semantic_tokens(
    text,
    checkpoint_path,
    temp_dir,
    prompt_tokens=None,
    prompt_text=None,
    device="cuda",
    compile_model=False,
    num_samples=1,
):
    semantic_tokens_path = os.path.join(temp_dir, "codes_0.npy")
    args = [
        "--text", text,
        "--checkpoint-path", checkpoint_path,
        "--device", device,
        "--num-samples", str(num_samples),
        "--output-dir", temp_dir
    ]

    if prompt_tokens:
        args.extend(["--prompt-tokens", prompt_tokens])

    if prompt_text:
        args.extend(["--prompt-text", prompt_text])

    if compile_model:
        args.append("--compile")

    with temporary_argv(["inference.py"] + args):
        try:
            text2semantic_main()
        except SystemExit:
            pass

    return semantic_tokens_path


def generate_speech_from_tokens(tokens_path, checkpoint_path, output_path="output.wav", device="cuda"):
    sys.argv = [
        "inference.py",
        "--input-path", tokens_path,
        "--output-path", output_path,
        "--checkpoint-path", checkpoint_path,
        "--device", device
    ]
    try:
        vqgan_inference.main()
    except SystemExit:
        pass
    return output_path


def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()


def text_to_speech(
    text,
    output_path="output.wav",
    reference_audio_path=None,
    checkpoint_dir="./models/fish-speech-1.5",
    device="cuda",
    compile_model=False
):
    start = time.time()
    llama_ckpt = checkpoint_dir
    decoder_ckpt = os.path.join(checkpoint_dir, "firefly-gan-vq-fsq-8x1024-21hz-generator.pth")

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            reference_tokens_path = None
            if reference_audio_path:
                reference_tokens_path = encode_reference_audio(reference_audio_path, temp_dir, device)

            semantic_tokens_path = generate_semantic_tokens(
                text=text,
                checkpoint_path=llama_ckpt,
                temp_dir=temp_dir,
                prompt_tokens=reference_tokens_path,
                device=device,
                compile_model=compile_model
            )

            generate_speech_from_tokens(
                tokens_path=semantic_tokens_path,
                checkpoint_path=decoder_ckpt,
                output_path=output_path,
                device=device
            )

            print(f"[INFO] Inference completed in {time.time() - start:.2f} seconds")
            return output_path

        except Exception as e:
            print(f"[ERROR] TTS failed: {e}")
            raise

        finally:
            clear_gpu_memory()
