import ctypes
from difflib import SequenceMatcher
import gc
import io
import json
import os
import re
import time
import urllib.request
import tkinter as tk
import numpy as np
from PIL import Image, ImageChops, ImageStat
import mss
import pygame
from rapidocr import RapidOCR, LangRec, LangDet, OCRVersion, ModelType

os.environ["OMP_NUM_THREADS"] = "2"

TTS_HOST = "127.0.0.1"
TTS_PORT = 8001
TTS_URL = "http" + "://" + TTS_HOST + ":" + str(TTS_PORT) + "/v1/audio/speech"

_no_proxy_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

VOICES = {
    0x70: ("xenia", "Ксения (женский)"),
    0x71: ("aidar", "Айдар (мужской)"),
    0x72: ("eugene", "Евгений (баритон)"),
    0x73: ("baya", "Байя (мягкий женский)"),
    0x74: ("kseniya", "Ксения-2 (спокойный)")
}
KEY_RESELECT = 0x77  # F8

current_voice = "xenia"
MAX_PRINT_WAIT = 3.5
STABLE_DELAY = 0.55
PIXEL_DIFF_THRESHOLD = 2.5
TEXT_SIMILARITY_RATIO = 0.88

pygame.mixer.init()
_current_sound = None

user32 = ctypes.windll.user32
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

_tk_root = tk.Tk()
_tk_root.withdraw()

def is_similar(a: str, b: str) -> bool:
    if not a or not b:
        return False
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= TEXT_SIMILARITY_RATIO

def select_region():
    _tk_root.update()
    win = tk.Toplevel(_tk_root)
    win.attributes("-fullscreen", True, "-alpha", 0.3, "-topmost", True)
    win.configure(bg="black")
    win.lift()
    win.focus_force()
    canvas = tk.Canvas(win, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    coords = {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
    rect = None

    def on_press(e):
        coords["x1"], coords["y1"] = e.x, e.y
        nonlocal rect
        rect = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="lime", width=2)

    def on_drag(e):
        if rect:
            canvas.coords(rect, coords["x1"], coords["y1"], e.x, e.y)

    def on_release(e):
        coords["x2"], coords["y2"] = e.x, e.y
        win.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    win.wait_window()
    _tk_root.update()

    left = min(coords["x1"], coords["x2"])
    top = min(coords["y1"], coords["y2"])
    width = max(16, abs(coords["x2"] - coords["x1"]))
    height = max(16, abs(coords["y2"] - coords["y1"]))
    return {"left": left, "top": top, "width": width, "height": height}

def speak(text: str):
    global _current_sound
    clean = re.sub(r'[^а-яА-ЯёЁ0-9\s.,!?-]', '', text).strip()
    if len(re.findall(r'[а-яА-ЯёЁ]', clean)) < 3:
        return

    payload = json.dumps({"input": clean, "voice": current_voice}).encode("utf-8")
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                TTS_URL,
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            audio_bytes = _no_proxy_opener.open(req, timeout=12).read()
            if audio_bytes and len(audio_bytes) > 44:
                if _current_sound is not None:
                    _current_sound.stop()
                snd = pygame.mixer.Sound(file=io.BytesIO(audio_bytes))
                snd.play()
                _current_sound = snd
            return
        except Exception as e:
            if attempt == 0:
                time.sleep(0.5)
            else:
                print(f"[Oshibka TTS ({TTS_URL})]: {e}", flush=True)

if __name__ == "__main__":
    print(f"TTS Endpoint: {TTS_URL}", flush=True)
    print("Zagruzka bystrogo detektora PP-OCRv5 Mobile...", flush=True)
    ocr = RapidOCR(params={
        "Global.use_cls": False,
        "Det.ocr_version": OCRVersion.PPOCRV5,
        "Det.lang_type": LangDet.MULTI,
        "Det.model_type": ModelType.MOBILE,
        "Rec.ocr_version": OCRVersion.PPOCRV5,
        "Rec.lang_type": LangRec.CYRILLIC,
        "Rec.model_type": ModelType.MOBILE
    })

    print("\n--- GORYACHIE KLAVISHI ---", flush=True)
    print("F1: Kseniya | F2: Aidar | F3: Eugene | F4: Baya | F5: Kseniya-2", flush=True)
    print("F8: Pereramka (vydelit novuyu oblast ekrana)\n", flush=True)

    print("Vydelite ramkoj nuzhnuyu oblast ekrana...", flush=True)
    region = select_region()
    print(f"Zona zafiksirovana: {region}. Nachinayu chtenie...", flush=True)

    with mss.mss() as sct:
        last_spoken = ""
        last_img = None
        change_started_at = None
        last_change_at = None

        while True:
            try:
                _tk_root.update()
            except Exception:
                pass

            for vk, (voice_id, label) in VOICES.items():
                if user32.GetAsyncKeyState(vk) & 0x0001:
                    if current_voice != voice_id:
                        current_voice = voice_id
                        print(f"[GOLOS IZMENEN]: {label}", flush=True)
                        if last_spoken:
                            speak(last_spoken)

            if user32.GetAsyncKeyState(KEY_RESELECT) & 0x0001:
                print("Vydelite novuyu oblast ekrana...", flush=True)
                if _current_sound is not None:
                    _current_sound.stop()
                region = select_region()
                last_spoken = ""
                last_img = None
                change_started_at = None
                last_change_at = None
                gc.collect()
                print(f"Novaya zona: {region}", flush=True)
                continue

            try:
                raw = sct.grab(region)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
            except Exception:
                time.sleep(0.5)
                continue

            now = time.time()

            if last_img is not None and last_img.size == img.size:
                diff = ImageChops.difference(img, last_img)
                stat = ImageStat.Stat(diff)
                mean_diff = sum(stat.mean) / len(stat.mean)
                if mean_diff > PIXEL_DIFF_THRESHOLD:
                    if change_started_at is None:
                        change_started_at = now
                    last_change_at = now
                    last_img = img
                    time.sleep(0.1)
                    continue

            last_img = img

            if change_started_at is not None:
                elapsed_total = now - change_started_at
                elapsed_stable = now - last_change_at

                if elapsed_stable >= STABLE_DELAY or elapsed_total >= MAX_PRINT_WAIT:
                    change_started_at = None
                    last_change_at = None

                    try:
                        img_np = np.array(img)
                        result = ocr(img_np, use_cls=False)
                        del img_np
                    except Exception as e:
                        print(f"[Oshibka OCR]: {e}", flush=True)
                        continue

                    current_text = ""
                    if result and result.txts:
                        current_text = " ".join(result.txts).strip()

                    if len(current_text) > 3 and not is_similar(current_text, last_spoken):
                        last_spoken = current_text
                        print(f"[{current_voice.upper()}]: {current_text}", flush=True)
                        speak(current_text)

            time.sleep(0.08)
