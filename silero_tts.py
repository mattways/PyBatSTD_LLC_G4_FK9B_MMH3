import gc
import io
import os
import re
import threading
import urllib.request
import wave
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_grad_enabled(False)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_FILE = "v4_ru.pt"
MODEL_URL = "https" + "://" + "models.silero.ai" + "/models/tts/ru/v4_ru.pt"

if not os.path.isfile(MODEL_FILE):
    print("Downloading Silero v4_ru.pt model...", flush=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)

device = torch.device("cpu")
model = torch.package.PackageImporter(MODEL_FILE).load_pickle("tts_models", "model")
model.to(device)
print("Loaded model: v4_ru.pt (Native Wave + Safe Mode)", flush=True)

tts_lock = threading.Lock()
VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
VALID_SPEAKERS = ("aidar", "baya", "kseniya", "xenia", "eugene")

class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: str = "xenia"
    speed: float = 1.18

def sanitize_ocr_words(text: str) -> str:
    words = text.split()
    safe_words = []
    for w in words:
        clean_w = re.sub(r'[^а-яА-ЯёЁ]', '', w)
        if not clean_w or len(clean_w) > 22:
            continue
        if len(clean_w) >= 2 and any(ch in VOWELS for ch in clean_w):
            safe_words.append(w)
        elif len(clean_w) == 1 and clean_w.lower() in ("я", "и", "а", "о", "у", "в", "к", "с"):
            safe_words.append(w)
    return " ".join(safe_words)

def split_text(text: str, max_len: int = 180):
    text = re.sub(r'[^а-яА-ЯёЁ\s.,!?]', ' ', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        s = sanitize_ocr_words(s.strip(" .,!?"))
        if len(re.findall(r'[а-яА-ЯёЁ]', s)) < 3:
            continue
        s = s + "."
        if len(current) + len(s) + 1 <= max_len:
            current = f"{current} {s}".strip()
        else:
            if current:
                chunks.append(current)
            if len(s) > max_len:
                words = s.split()
                part = ""
                for w in words:
                    if len(part) + len(w) + 1 <= max_len:
                        part = f"{part} {w}".strip()
                    else:
                        if len(re.findall(r'[а-яА-ЯёЁ]', part)) >= 3:
                            chunks.append(part.rstrip(".,!?") + ".")
                        part = w
                current = part if len(re.findall(r'[а-яА-ЯёЁ]', part)) >= 3 else ""
            else:
                current = s
    if current and len(re.findall(r'[а-яА-ЯёЁ]', current)) >= 3:
        chunks.append(current)
    return chunks

def trim_silence(audio_np: np.ndarray, threshold: float = 0.015) -> np.ndarray:
    non_silent = np.flatnonzero(np.abs(audio_np) > threshold)
    if non_silent.size == 0:
        return audio_np
    first_idx = int(non_silent.item(0))
    last_idx = int(non_silent.item(-1))
    return audio_np[max(0, first_idx - 1000):min(len(audio_np), last_idx + 1000)]

def encode_native_wav(audio_np: np.ndarray, rate: int) -> bytes:
    pcm_s16 = np.clip(audio_np * 32767.0, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_s16.tobytes())
    return buf.getvalue()

@app.post("/v1/audio/speech")
def generate_speech(req: SpeechRequest):
    sample_rate = 24000
    custom_voice = None
    for ext in (".pt", ".pth"):
        candidate = os.path.join("samples", f"{req.voice}{ext}")
        if os.path.exists(candidate):
            custom_voice = candidate
            break

    chunks = split_text(req.input)
    if not chunks:
        return Response(content=b"", media_type="audio/wav")

    audio_parts = []
    speaker = req.voice if req.voice in VALID_SPEAKERS else "xenia"

    with tts_lock:
        for chunk in chunks:
            try:
                if custom_voice:
                    audio = model.apply_tts(
                        text=chunk,
                        speaker="random",
                        voice_path=custom_voice,
                        sample_rate=sample_rate
                    )
                else:
                    audio = model.apply_tts(
                        text=chunk,
                        speaker=speaker,
                        sample_rate=sample_rate
                    )
                audio_parts.append(trim_silence(audio.detach().cpu().numpy().copy()))
                del audio
            except Exception as e:
                print(f"[Skip chunk]: {chunk} -> {e}", flush=True)

        if not audio_parts:
            gc.collect()
            return Response(content=b"", media_type="audio/wav")

        full_audio = np.concatenate(audio_parts)
        wav_bytes = encode_native_wav(full_audio, int(sample_rate * 1.18))
        print(f"[OK] Generated {len(wav_bytes)} bytes", flush=True)
        del full_audio, audio_parts
        gc.collect()

    return Response(content=wav_bytes, media_type="audio/wav")

if __name__ == "__main__":
    os.makedirs("samples", exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8001)
