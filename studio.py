import sys
import gc
import io
import os
import json
import time
import socket
import base64
import threading
import subprocess
from datetime import datetime
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import requests
from PIL import Image, ImageOps

SD_URL = "http://127.0.0.1:8081"
WEB_HOST = "0.0.0.0"
WEB_PORT = 8082

OUT_IMG_DIR = Path("output_img")
OUT_VID_DIR = Path("output_vid")
LOG_FILE_PATH = Path("sd_server.log")
OUT_IMG_DIR.mkdir(exist_ok=True)
OUT_VID_DIR.mkdir(exist_ok=True)

# === MODEL FILES ===
IMG_DIFF = "Master_Flux2Klein_v4_Turbo_Q6_K.gguf"
IMG_LLM = "Qwen3-VL-8B-Q4_K_M.gguf"
IMG_VAE = "flux2-vae.safetensors"

VID_DIFF = "minimax_h3_fl2va_hyperflow_flash_14b-Q3_K_M.gguf"
VID_VAE = "minimax_h3_sol_vae_fp16.safetensors"
VID_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
VID_LLM = "Qwen3-VL-32B-H3-L0-49-IQ2_XXS.gguf"
VID_VISION = "Qwen3-VL-32B-Instruct.mmproj-Q8_0.gguf"

http = requests.Session()
GEN_LOCK = threading.Lock()
PROC_LOCK = threading.Lock()
SD_PROC = None
SD_LOG_HANDLE = None

STATE = {
    "active_mode": "none",
    "busy": False,
    "switching": False,
    "status": "Idle (0 GB VRAM used)",
    "last_file": "",
    "last_type": "",
    "last_info": ""
}


def get_strict_env():
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    return env


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_backend():
    global SD_PROC, SD_LOG_HANDLE
    if SD_PROC is not None:
        try:
            SD_PROC.terminate()
            SD_PROC.wait(timeout=3)
        except Exception:
            try:
                SD_PROC.kill()
                SD_PROC.wait(timeout=2)
            except Exception:
                pass
        SD_PROC = None
    if SD_LOG_HANDLE is not None:
        try:
            SD_LOG_HANDLE.close()
        except Exception:
            pass
        SD_LOG_HANDLE = None
    subprocess.run(
        ["taskkill", "/IM", "sd-server.exe", "/F", "/T"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    for _ in range(25):
        if not is_port_in_use(8081):
            break
        time.sleep(0.2)
    time.sleep(0.5)


def check_files_exist(files):
    for f in files:
        if not Path(f).is_file():
            raise FileNotFoundError(f"Missing model file: {f}")


def switch_backend_mode(target_mode: str):
    global SD_PROC, SD_LOG_HANDLE
    if STATE["busy"]:
        raise RuntimeError("Cannot switch models while generation is running")

    with PROC_LOCK:
        STATE["switching"] = True
        try:
            STATE["status"] = "Unloading previous model..."
            kill_backend()
            STATE["active_mode"] = "none"

            if target_mode == "none":
                STATE["status"] = "Unloaded (0 GB VRAM used)"
                return

            if target_mode == "image":
                check_files_exist([IMG_DIFF, IMG_LLM, IMG_VAE])
                cmd = [
                    "sd-server.exe",
                    "--diffusion-model", IMG_DIFF,
                    "--llm", IMG_LLM,
                    "--vae", IMG_VAE,
                    "--lora-model-dir", "loras",
                    "--auto-fit", "off",
                    "--backend", "all=cuda0,te=cuda1,vae=cuda0",
                    "-t", "2",
                    "--flash-attn", "off",
                    "--rng", "cuda",
                    "--sampling-method", "euler",
                    "--cfg-scale", "1.0",
                    "--listen-ip", "127.0.0.1",
                    "--listen-port", "8081"
                ]
            elif target_mode == "video":
                check_files_exist([VID_DIFF, VID_VAE, VID_LLM, VID_VISION])
                cmd = [
                    "sd-server.exe",
                    "--diffusion-model", VID_DIFF,
                    "--vae", VID_VAE,
                    "--llm", VID_LLM,
                    "--llm_vision", VID_VISION,
                    "--auto-fit", "off",
                    "--backend", "all=cuda0,te=cuda1,vision=cuda1,vae=cuda0",
                    "-t", "2",
                    "--flash-attn", "off",
                    "--rng", "cuda",
                    "--model-args", "minimax_h3_attn_chunk_budget=0.05",
                    "--sampling-method", "euler",
                    "--cfg-scale", "1.0",
                    "--listen-ip", "127.0.0.1",
                    "--listen-port", "8081"
                ]
            elif target_mode == "audio":
                check_files_exist([VID_DIFF, VID_VAE, VID_AUDIO, VID_LLM, VID_VISION])
                cmd = [
                    "sd-server.exe",
                    "--diffusion-model", VID_DIFF,
                    "--vae", VID_VAE,
                    "--audio-vae", VID_AUDIO,
                    "--llm", VID_LLM,
                    "--llm_vision", VID_VISION,
                    "--auto-fit", "off",
                    "--backend", "all=cuda0,te=cuda1,vision=cuda1,vae=cuda0",
                    "-t", "2",
                    "--flash-attn", "off",
                    "--rng", "cuda",
                    "--model-args", "minimax_h3_attn_chunk_budget=0.05",
                    "--sampling-method", "euler",
                    "--cfg-scale", "1.0",
                    "--listen-ip", "127.0.0.1",
                    "--listen-port", "8081"
                ]
            else:
                raise ValueError(f"Unknown mode: {target_mode}")

            STATE["active_mode"] = target_mode
            STATE["status"] = f"Loading {target_mode.upper()} into VRAM..."
            SD_LOG_HANDLE = open(LOG_FILE_PATH, "w", encoding="utf-8", errors="replace", buffering=1)
            SD_PROC = subprocess.Popen(
                cmd,
                env=get_strict_env(),
                stdout=SD_LOG_HANDLE,
                stderr=subprocess.STDOUT,
                text=True
            )
        finally:
            STATE["switching"] = False


def prepare_image_ref_b64(data_url: str, max_side: int = 1024):
    raw_b64 = data_url.split(",")[-1]
    raw_bytes = base64.b64decode(raw_b64)
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        w, h = img.size
        scale = min(max_side / max(w, h), 1.0)
        nw = max(256, int(w * scale) // 16 * 16)
        nh = max(256, int(h * scale) // 16 * 16)
        if (nw, nh) != (w, h):
            img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return b64, nw, nh


def prepare_video_frame_b64(data_url: str, target_w: int, target_h: int):
    raw_b64 = data_url.split(",")[-1]
    raw_bytes = base64.b64decode(raw_b64)
    with Image.open(io.BytesIO(raw_bytes)) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        fitted = ImageOps.fit(img, (target_w, target_h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
        buf = io.BytesIO()
        fitted.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        return b64


def read_last_log_lines(max_chars: int = 350) -> str:
    try:
        if LOG_FILE_PATH.is_file():
            content = LOG_FILE_PATH.read_text(encoding="utf-8", errors="replace").strip()
            return content[-max_chars:]
    except Exception:
        pass
    return ""


def get_sd_status():
    if STATE["switching"]:
        return {"state": "switching"}
    if STATE["active_mode"] == "none":
        return {"state": "unloaded"}
    if SD_PROC is not None and SD_PROC.poll() is not None:
        err_out = read_last_log_lines(350)
        return {"state": "crashed", "error": err_out}
    try:
        r = http.get(f"{SD_URL}/sdcpp/v1/models/status", timeout=2)
        if r.status_code == 200:
            data = r.json()
            if data.get("state") == "ready" and not STATE["busy"] and STATE["status"].startswith("Loading "):
                STATE["status"] = "Ready (Strict VRAM)"
            return data
    except Exception:
        pass
    return {"state": "loading"}


def get_loras():
    if STATE["active_mode"] != "image":
        return []
    try:
        r = http.get(f"{SD_URL}/sdcpp/v1/capabilities", timeout=3)
        if r.status_code == 200:
            return r.json().get("loras", [])
    except Exception:
        pass
    return []


def poll_job(resp_data: dict) -> dict:
    poll_url = resp_data.get("poll_url") or f"/sdcpp/v1/jobs/{resp_data['id']}"
    errors = 0
    while True:
        if SD_PROC is not None and SD_PROC.poll() is not None:
            log_tail = read_last_log_lines(300)
            raise RuntimeError(f"sd-server.exe terminated: {log_tail}")
        try:
            r = http.get(f"{SD_URL}{poll_url}", timeout=10)
            if r.status_code != 200:
                time.sleep(1.0)
                continue
            job = r.json()
            errors = 0
        except requests.RequestException as e:
            errors += 1
            if errors > 15:
                raise RuntimeError(f"Server connection lost: {e}")
            time.sleep(2.0)
            continue

        st = job.get("status")
        if st == "completed":
            return job["result"]
        if st in ("failed", "cancelled", "error"):
            err = job.get("error") or {}
            msg = err.get("message", err) if isinstance(err, dict) else str(err)
            log_tail = read_last_log_lines(250)
            raise RuntimeError(f"{msg} | LOG: {log_tail}" if log_tail else msg)

        prog = job.get("progress") or {}
        if prog:
            stage = prog.get("stage", st)
            comp, tot = prog.get("completed"), prog.get("total")
            if comp is not None and tot:
                STATE["status"] = f"{stage} ({comp}/{tot})"
            else:
                STATE["status"] = f"{stage}..."
        time.sleep(1.0)


def generate_image(data: dict):
    quality = data.get("quality", "fast")
    aspect = data.get("aspect", "9:16")
    raw_prompt = data.get("prompt", "").strip()
    use_raw = data.get("raw_prompt", False)
    ref_b64_in = data.get("ref_image")
    lora_path = data.get("lora_path", "")
    lora_mult = float(data.get("lora_mult", 1.0))

    edit_on = bool(ref_b64_in)
    if edit_on:
        steps = 4
        cfg = 2.0
        neg = "Distorted anatomy, changed facial identity, blurry details, artifacts."
    else:
        if quality == "max":
            steps = 8
            cfg = 1.5
            neg = "Overly smooth plastic skin, airbrushed face, CGI render look, distorted hands, extra fingers, blurry details."
        else:
            steps = 4
            cfg = 1.0
            neg = ""

    sizes = {"9:16": (768, 1344), "1:1": (1024, 1024), "16:9": (1344, 768)}
    w, h = sizes.get(aspect, (768, 1344))

    suffix = ". Medium shot candid photograph in soft natural lighting, balanced framing, realistic skin texture and natural colors."
    prompt = raw_prompt if (use_raw or edit_on) else (raw_prompt.rstrip(".!") + suffix)

    loras = [{"path": lora_path, "multiplier": lora_mult}] if lora_path else []
    payload = {
        "prompt": prompt,
        "width": w,
        "height": h,
        "seed": -1,
        "preview": "none",
        "lora": loras,
        "sample_params": {
            "sample_steps": steps,
            "sample_method": "euler",
            "guidance": {"txt_cfg": cfg},
        },
        "vae_tiling_params": {"enabled": False},
        "output_format": "png",
    }
    if neg:
        payload["negative_prompt"] = neg
    if edit_on:
        ref_b64, rw, rh = prepare_image_ref_b64(ref_b64_in, max_side=1024)
        payload.update({"ref_images": [ref_b64], "width": rw, "height": rh})

    t0 = time.time()
    STATE["status"] = "Sending request..."
    r = http.post(f"{SD_URL}/sdcpp/v1/img_gen", json=payload, timeout=60)
    del payload
    if r.status_code not in (200, 202):
        raise RuntimeError(f"Error {r.status_code}: {r.text}")

    resp_json = r.json()
    imgs = poll_job(resp_json)["images"] if r.status_code == 202 else resp_json["images"]
    item = imgs[0] if isinstance(imgs, list) else imgs
    b64 = item["b64_json"].split(",")[-1] if isinstance(item, dict) else item.split(",")[-1]
    seed = item.get("generation_info", {}).get("seed", -1) if isinstance(item, dict) else -1

    tag = "edit" if edit_on else quality
    fname = f"klein_{tag}_{datetime.now():%Y%m%d_%H%M%S}.png"
    out = OUT_IMG_DIR / fname
    out.write_bytes(base64.b64decode(b64))
    elapsed = time.time() - t0

    STATE["last_file"] = f"/files/img/{fname}"
    STATE["last_type"] = "image"
    STATE["last_info"] = f"{fname} | seed: {seed} | {elapsed:.1f}s"
    del imgs, item, b64, resp_json
    gc.collect()


def generate_video(data: dict, audio_on: bool):
    quality = data.get("quality", "fast")
    prompt = data.get("prompt", "").strip()
    start_img = data.get("start_image")

    if quality == "max":
        res_w, res_h = 480, 864
        frames = 39
        steps = 8
        sigmas_preset = "hyperflow-h3-flash-8"
    else:
        res_w, res_h = 384, 640
        frames = 39
        steps = 6
        sigmas_preset = "hyperflow-h3-flash-6"

    out_fmt = "webm"
    payload = {
        "prompt": prompt,
        "negative_prompt": "",
        "width": res_w,
        "height": res_h,
        "video_frames": frames,
        "fps": 24,
        "seed": -1,
        "preview": "none",
        "sample_params": {
            "sample_method": "euler",
            "sample_steps": steps,
            "sigmas": sigmas_preset,
            "guidance": {"txt_cfg": 1.0},
        },
        "vae_tiling_params": {
            "enabled": True,
            "tile_size_w": 256,
            "tile_size_h": 256,
            "temporal_tiling": True,
            "temporal_tile_frames": 10,
            "temporal_tile_overlap": 5,
        },
        "output_format": out_fmt,
    }
    if start_img:
        payload["init_image"] = prepare_video_frame_b64(start_img, res_w, res_h)

    t0 = time.time()
    STATE["status"] = "Queuing video job..."
    resp = http.post(f"{SD_URL}/sdcpp/v1/vid_gen", json=payload, timeout=30)
    if resp.status_code == 400 and out_fmt == "webm":
        out_fmt = "avi"
        payload["output_format"] = "avi"
        resp = http.post(f"{SD_URL}/sdcpp/v1/vid_gen", json=payload, timeout=30)

    del payload
    gc.collect()

    if resp.status_code != 202:
        raise RuntimeError(f"Error {resp.status_code}: {resp.text}")

    job_data = resp.json()
    res = poll_job(job_data)
    ext = res.get("output_format", out_fmt)
    prefix = "av" if audio_on else "vid"
    fname = f"{prefix}_{quality}_{datetime.now():%Y%m%d_%H%M%S}.{ext}"
    out = OUT_VID_DIR / fname
    out.write_bytes(base64.b64decode(res["b64_json"]))
    elapsed = time.time() - t0
    cnt = res.get("frame_count", frames)

    STATE["last_file"] = f"/files/vid/{fname}"
    STATE["last_type"] = "video"
    STATE["last_info"] = f"{fname} | {res_w}x{res_h} | {cnt} frames | {elapsed:.1f}s"
    del res, job_data
    gc.collect()


def list_gallery():
    items = []
    for folder, ftype, prefix in [(OUT_IMG_DIR, "image", "/files/img/"), (OUT_VID_DIR, "video", "/files/vid/")]:
        if folder.exists():
            for p in folder.iterdir():
                if p.is_file() and not p.name.startswith("."):
                    st = p.stat()
                    items.append({
                        "name": p.name,
                        "type": ftype,
                        "url": prefix + p.name,
                        "mtime": st.st_mtime,
                        "size_mb": round(st.st_size / (1024 * 1024), 2)
                    })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:100]


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Render Studio (LAN)</title>
<style>
  body { font-family: system-ui, sans-serif; background: #121212; color: #eee; max-width: 820px; margin: 0 auto; padding: 16px; }
  .tabs { display: flex; gap: 8px; margin-bottom: 14px; }
  .tab-btn {
    flex: 1; padding: 12px; background: #1e1e1e; color: #aaa; border: 1px solid #333;
    border-radius: 8px; font-size: 1rem; font-weight: bold; cursor: pointer;
  }
  .tab-btn.active { background: #2563eb; color: #fff; border-color: #2563eb; }
  .card { background: #1e1e1e; border: 1px solid #333; border-radius: 10px; padding: 16px; margin-bottom: 16px; }
  h2 { margin-top: 0; font-size: 1.15rem; color: #4ade80; }
  label { display: block; margin: 10px 0 4px; font-size: 0.9rem; color: #aaa; }
  textarea, select, input[type="number"], input[type="file"] {
    width: 100%; box-sizing: border-box; background: #2a2a2a; color: #fff; border: 1px solid #444;
    border-radius: 6px; padding: 8px; font-size: 0.95rem;
  }
  .row { display: flex; gap: 10px; flex-wrap: wrap; }
  .col { flex: 1; min-width: 150px; }
  .engine-bar { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
  .eng-btn {
    flex: 1; min-width: 110px; padding: 10px 8px; background: #252525; color: #ccc;
    border: 2px solid #383838; border-radius: 8px; font-weight: bold; cursor: pointer; font-size: 0.9rem;
  }
  .eng-btn.active { border-color: #16a34a; background: #14532d; color: #fff; }
  .eng-btn.unload { border-color: #7f1d1d; color: #fca5a5; max-width: 140px; }
  .eng-btn.unload:hover { background: #7f1d1d; color: #fff; }
  .eng-btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .mode-cards { display: flex; gap: 10px; margin: 10px 0; }
  .mode-card {
    flex: 1; padding: 12px; background: #252525; border: 2px solid #383838;
    border-radius: 8px; cursor: pointer; text-align: left; color: #ddd;
  }
  .mode-card.selected { border-color: #2563eb; background: #1e293b; color: #fff; }
  .mode-card b { display: block; font-size: 1rem; margin-bottom: 4px; }
  .mode-card span { font-size: 0.82rem; color: #9ca3af; }
  button.primary {
    width: 100%; background: #2563eb; color: #fff; border: none; border-radius: 8px;
    padding: 13px; font-size: 1rem; font-weight: bold; cursor: pointer; margin-top: 14px;
  }
  button.primary:disabled { background: #444; cursor: not-allowed; }
  .status { padding: 10px; background: #262626; border-radius: 6px; font-family: monospace; margin-bottom: 12px; }
  img, video { max-width: 100%; border-radius: 8px; margin-top: 10px; background: #000; display: block; }
  .preview-thumb { max-height: 200px; width: auto; margin-top: 8px; border: 1px solid #444; }
  .hidden { display: none !important; }
  .gal-filters { display: flex; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
  .gal-filters button {
    padding: 8px 14px; background: #2a2a2a; color: #eee; border: 1px solid #444;
    border-radius: 6px; cursor: pointer; font-size: 0.9rem;
  }
  .gal-filters button.active { background: #16a34a; border-color: #16a34a; color: #fff; }
  .gal-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
  .gal-item { background: #252525; border: 1px solid #383838; border-radius: 8px; padding: 10px; display: flex; flex-direction: column; justify-content: space-between; }
  .gal-item img, .gal-item video { width: 100%; height: 260px; object-fit: cover; margin-top: 0; border-radius: 6px; cursor: pointer; }
  .gal-meta { font-size: 0.78rem; color: #aaa; margin-top: 8px; word-break: break-all; }
  .btn-group { display: flex; gap: 6px; margin-top: 8px; }
  .dl-btn {
    flex: 1; text-align: center; padding: 7px; background: #333; color: #fff; text-decoration: none;
    border-radius: 5px; font-size: 0.85rem;
  }
  .del-btn {
    padding: 7px 12px; background: #991b1b; color: #fff; border: none; border-radius: 5px;
    font-size: 0.85rem; cursor: pointer;
  }
  .file-box { background: #252525; border: 1px dashed #555; border-radius: 8px; padding: 10px; margin-top: 10px; }
  .clear-file-btn {
    background: #444; color: #fff; border: none; border-radius: 5px; padding: 5px 10px;
    font-size: 0.8rem; cursor: pointer; margin-top: 6px;
  }
</style>
</head>
<body>

<div class="tabs">
  <button id="tabGenBtn" class="tab-btn active" onclick="switchTab('gen')">Generator</button>
  <button id="tabGalBtn" class="tab-btn" onclick="switchTab('gal')">Gallery</button>
</div>

<div id="viewGen">
  <div class="card">
    <h2>Active Model (Strict 20GB VRAM - No RAM/CPU)</h2>
    <div class="engine-bar">
      <button id="engImage" class="eng-btn" onclick="selectEngine('image')">IMAGE (Flux.2)</button>
      <button id="engVideo" class="eng-btn" onclick="selectEngine('video')">VIDEO (Silent)</button>
      <button id="engAudio" class="eng-btn" onclick="selectEngine('audio')">VIDEO + AUDIO</button>
      <button id="engNone" class="eng-btn unload" onclick="selectEngine('none')">Free VRAM</button>
    </div>
    <div id="statusBox" class="status">Select a model above to load into VRAM...</div>
  </div>

  <div id="workspace" class="card hidden">
    <label>Quality Mode (Strictly inside 20 GB VRAM)</label>
    <div class="mode-cards">
      <div id="qFastBtn" class="mode-card selected" onclick="setQuality('fast')">
        <b>1. Fast Draft</b>
        <span id="qFastDesc">6 steps</span>
      </div>
      <div id="qMaxBtn" class="mode-card" onclick="setQuality('max')">
        <b>2. Max Quality</b>
        <span id="qMaxDesc">8 steps</span>
      </div>
    </div>

    <div id="imgControls">
      <div class="row">
        <div class="col">
          <label>Aspect Ratio</label>
          <select id="imgAspect">
            <option value="9:16" selected>9:16 Mobile Portrait (768x1344)</option>
            <option value="1:1">1:1 Square (1024x1024)</option>
            <option value="16:9">16:9 Landscape (1344x768)</option>
          </select>
        </div>
        <div class="col">
          <label>LoRA</label>
          <select id="loraSelect">
            <option value="">No LoRA</option>
          </select>
        </div>
        <div class="col" style="max-width: 110px;">
          <label>LoRA Weight</label>
          <input type="number" id="loraMult" value="1.0" step="0.05" min="-2" max="2">
        </div>
      </div>

      <div class="file-box">
        <label style="margin-top:0;">Reference Image (Optional, auto-switches to EDIT 4 steps / CFG 2.0)</label>
        <input type="file" id="imgRef" accept="image/*" onchange="previewFile(this, 'imgRefPrev')">
        <img id="imgRefPrev" class="preview-thumb hidden">
        <button id="imgRefClear" class="clear-file-btn hidden" onclick="clearFile('imgRef', 'imgRefPrev')">Remove Image</button>
      </div>

      <label style="margin-top: 12px; cursor: pointer;">
        <input type="checkbox" id="rawPrompt"> Raw prompt (disable automatic photo suffix)
      </label>
    </div>

    <div id="vidControls" class="hidden">
      <div class="file-box">
        <label style="margin-top:0;">Start Image (Optional: auto-cropped to 9:16 portrait; leave empty for Text-to-Video)</label>
        <input type="file" id="vidStartImg" accept="image/*" onchange="previewFile(this, 'vidStartPrev')">
        <img id="vidStartPrev" class="preview-thumb hidden">
        <button id="vidStartImgClear" class="clear-file-btn hidden" onclick="clearFile('vidStartImg', 'vidStartPrev')">Remove Image</button>
      </div>
    </div>

    <label>Prompt</label>
    <textarea id="prompt" rows="3" placeholder="Enter your prompt..."></textarea>

    <button id="genBtn" class="primary" onclick="startGen()">Generate</button>
  </div>

  <div id="resultCard" class="card" style="display:none;">
    <div id="resultInfo" class="status" data-file=""></div>
    <div id="mediaBox"></div>
    <div id="resActions" class="btn-group"></div>
  </div>
</div>

<div id="viewGal" class="hidden">
  <div class="card">
    <h2>File Gallery (output_img / output_vid)</h2>
    <div class="gal-filters">
      <button id="fltAll" class="active" onclick="setFilter('all')">All</button>
      <button id="fltImg" onclick="setFilter('image')">Images</button>
      <button id="fltVid" onclick="setFilter('video')">Videos</button>
      <button onclick="loadGallery()">Refresh</button>
    </div>
    <div id="galGrid" class="gal-grid">Loading...</div>
  </div>
</div>

<script>
let currentMode = "none";
let currentQuality = "fast";
let lorasLoadedForMode = "";
let galItems = [];
let galFilter = "all";
let isSwitching = false;

function setQuality(q) {
  currentQuality = q;
  document.getElementById("qFastBtn").classList.toggle("selected", q === "fast");
  document.getElementById("qMaxBtn").classList.toggle("selected", q === "max");
}

function previewFile(input, imgId) {
  const img = document.getElementById(imgId);
  const clr = document.getElementById(input.id + "Clear");
  if (input.files && input.files[0]) {
    const r = new FileReader();
    r.onload = () => {
      img.src = r.result;
      img.classList.remove("hidden");
      if (clr) clr.classList.remove("hidden");
    };
    r.readAsDataURL(input.files[0]);
  } else {
    img.src = "";
    img.classList.add("hidden");
    if (clr) clr.classList.add("hidden");
  }
}

function clearFile(inputId, imgId) {
  const inp = document.getElementById(inputId);
  const img = document.getElementById(imgId);
  const clr = document.getElementById(inputId + "Clear");
  inp.value = "";
  img.src = "";
  img.classList.add("hidden");
  if (clr) clr.classList.add("hidden");
}

function switchTab(tab) {
  document.getElementById("viewGen").classList.toggle("hidden", tab !== "gen");
  document.getElementById("viewGal").classList.toggle("hidden", tab !== "gal");
  document.getElementById("tabGenBtn").classList.toggle("active", tab === "gen");
  document.getElementById("tabGalBtn").classList.toggle("active", tab === "gal");
  if (tab === "gal") loadGallery();
}

function setFilter(flt) {
  galFilter = flt;
  document.getElementById("fltAll").classList.toggle("active", flt === "all");
  document.getElementById("fltImg").classList.toggle("active", flt === "image");
  document.getElementById("fltVid").classList.toggle("active", flt === "video");
  renderGallery();
}

async function selectEngine(mode) {
  if (isSwitching) return;
  isSwitching = true;
  lorasLoadedForMode = "";
  document.getElementById("engImage").disabled = true;
  document.getElementById("engVideo").disabled = true;
  document.getElementById("engAudio").disabled = true;
  document.getElementById("engNone").disabled = true;
  document.getElementById("genBtn").disabled = true;
  try {
    await fetch("/api/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode })
    });
  } finally {
    isSwitching = false;
    pollState();
  }
}

async function refreshLoras() {
  try {
    const res = await fetch("/api/loras");
    const loras = await res.json();
    const sel = document.getElementById("loraSelect");
    sel.innerHTML = '<option value="">No LoRA</option>';
    loras.forEach(l => {
      const opt = document.createElement("option");
      opt.value = l.path;
      opt.innerText = l.display_path || l.name;
      sel.appendChild(opt);
    });
  } catch (e) {}
}

async function deleteFile(url, name) {
  if (!confirm("Delete file " + name + " permanently?")) return;
  try {
    const res = await fetch("/api/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url })
    });
    if (res.ok) {
      if (document.getElementById("resultInfo").dataset.file === url) {
        document.getElementById("resultCard").style.display = "none";
        document.getElementById("resultInfo").dataset.file = "";
      }
      loadGallery();
    } else {
      alert("Failed to delete file");
    }
  } catch (e) {
    alert("Delete error: " + e);
  }
}

async function loadGallery() {
  try {
    const res = await fetch("/api/gallery");
    galItems = await res.json();
    renderGallery();
  } catch (e) {
    document.getElementById("galGrid").innerText = "Error loading gallery";
  }
}

function renderGallery() {
  const grid = document.getElementById("galGrid");
  const filtered = galItems.filter(x => galFilter === "all" || x.type === galFilter);
  if (!filtered.length) {
    grid.innerHTML = "No files yet";
    return;
  }
  grid.innerHTML = filtered.map(item => {
    const media = item.type === "image"
      ? `<img src="${item.url}" loading="lazy" onclick="window.open('${item.url}', '_blank')">`
      : `<video src="${item.url}" controls preload="metadata"></video>`;
    return `<div class="gal-item">${media}<div><div class="gal-meta">${item.name} (${item.size_mb} MB)</div><div class="btn-group"><a class="dl-btn" href="${item.url}" download="${item.name}">Download</a><button class="del-btn" onclick="deleteFile('${item.url}', '${item.name}')">Delete</button></div></div></div>`;
  }).join("");
}

function readFileB64(input) {
  return new Promise((resolve) => {
    if (!input.files || !input.files[0]) return resolve(null);
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.readAsDataURL(input.files[0]);
  });
}

function updateModeUI(mode) {
  currentMode = mode;
  document.getElementById("engImage").classList.toggle("active", mode === "image");
  document.getElementById("engVideo").classList.toggle("active", mode === "video");
  document.getElementById("engAudio").classList.toggle("active", mode === "audio");
  const ws = document.getElementById("workspace");
  if (mode === "none") {
    ws.classList.add("hidden");
    return;
  }
  ws.classList.remove("hidden");
  if (mode === "image") {
    document.getElementById("imgControls").classList.remove("hidden");
    document.getElementById("vidControls").classList.add("hidden");
    document.getElementById("qFastDesc").innerText = "4 steps, CFG 1.0 - fast draft (9:16 default)";
    document.getElementById("qMaxDesc").innerText = "8 steps, CFG 1.5 + Neg - detailed skin texture";
  } else {
    document.getElementById("imgControls").classList.add("hidden");
    document.getElementById("vidControls").classList.remove("hidden");
    document.getElementById("qFastDesc").innerText = "6 steps, 384x640 (9:16), 39 frames (1.6s) - native flash draft";
    document.getElementById("qMaxDesc").innerText = "8 steps, 480x864 (9:16), 39 frames (1.6s) - crisp high-res in 20GB VRAM";
  }
}

async function pollState() {
  try {
    const res = await fetch("/api/state");
    const st = await res.json();
    if (st.active_mode !== currentMode) {
      updateModeUI(st.active_mode);
    }
    const sdState = st.sd.state;
    const sdReady = sdState === "ready";
    const lockButtons = st.busy || st.switching || isSwitching;
    document.getElementById("engImage").disabled = lockButtons;
    document.getElementById("engVideo").disabled = lockButtons;
    document.getElementById("engAudio").disabled = lockButtons;
    document.getElementById("engNone").disabled = lockButtons || st.active_mode === "none";
    document.getElementById("genBtn").disabled = lockButtons || !sdReady;
    if (sdReady && st.active_mode === "image" && lorasLoadedForMode !== "image") {
      lorasLoadedForMode = "image";
      refreshLoras();
    }
    const errSuffix = st.sd.error ? ` | ERR: ${st.sd.error}` : "";
    document.getElementById("statusBox").innerText =
      `Model: ${st.active_mode.toUpperCase()} (${sdState.toUpperCase()}) | Task: ${st.status}${errSuffix}`;
    if (st.last_file && document.getElementById("resultInfo").dataset.file !== st.last_file) {
      document.getElementById("resultCard").style.display = "block";
      document.getElementById("resultInfo").dataset.file = st.last_file;
      document.getElementById("resultInfo").innerText = st.last_info;
      const box = document.getElementById("mediaBox");
      if (st.last_type === "image") {
        box.innerHTML = `<img src="${st.last_file}">`;
      } else {
        box.innerHTML = `<video src="${st.last_file}" controls autoplay loop></video>`;
      }
      const fname = st.last_file.split("/").pop();
      document.getElementById("resActions").innerHTML = `<a class="dl-btn" href="${st.last_file}" download="${fname}">Download Result</a> <button class="del-btn" onclick="deleteFile('${st.last_file}', '${fname}')">Delete</button>`;
    }
  } catch (e) {}
}

async function startGen() {
  const prompt = document.getElementById("prompt").value.trim();
  if (!prompt) return alert("Enter a prompt!");
  let payload = { prompt, quality: currentQuality };
  if (currentMode === "image") {
    payload.aspect = document.getElementById("imgAspect").value;
    payload.lora_path = document.getElementById("loraSelect").value;
    payload.lora_mult = document.getElementById("loraMult").value;
    payload.raw_prompt = document.getElementById("rawPrompt").checked;
    payload.ref_image = await readFileB64(document.getElementById("imgRef"));
  } else {
    payload.start_image = await readFileB64(document.getElementById("vidStartImg"));
  }
  document.getElementById("genBtn").disabled = true;
  await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

pollState();
setInterval(pollState, 1500);
</script>
</body>
</html>
"""


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            data = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/api/loras":
            self._send_json(get_loras())
        elif path == "/api/state":
            sd_info = get_sd_status()
            self._send_json({**STATE, "sd": sd_info})
        elif path == "/api/gallery":
            self._send_json(list_gallery())
        elif path.startswith("/files/img/") or path.startswith("/files/vid/"):
            is_img = path.startswith("/files/img/")
            base_dir = OUT_IMG_DIR if is_img else OUT_VID_DIR
            fname = os.path.basename(path)
            fpath = base_dir / fname
            if fpath.is_file():
                ext = fpath.suffix.lower()
                mime = {
                    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                    ".webm": "video/webm", ".avi": "video/x-msvideo"
                }.get(ext, "application/octet-stream")
                content = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/switch":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            target = body.get("mode", "none")

            def switcher():
                try:
                    switch_backend_mode(target)
                except Exception as e:
                    STATE["status"] = f"Switch error: {e}"

            threading.Thread(target=switcher, daemon=True).start()
            self._send_json({"ok": True})

        elif path == "/api/generate":
            if STATE["busy"] or STATE["switching"]:
                return self._send_json({"error": "Busy"}, 409)
            if STATE["active_mode"] == "none":
                return self._send_json({"error": "No model loaded"}, 400)

            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            mode_now = STATE["active_mode"]

            def worker():
                with GEN_LOCK:
                    STATE["busy"] = True
                    try:
                        if mode_now == "image":
                            generate_image(body)
                        elif mode_now == "video":
                            generate_video(body, audio_on=False)
                        elif mode_now == "audio":
                            generate_video(body, audio_on=True)
                        STATE["status"] = "Done"
                    except Exception as e:
                        STATE["status"] = f"Error: {e}"
                    finally:
                        STATE["busy"] = False

            threading.Thread(target=worker, daemon=True).start()
            self._send_json({"ok": True})

        elif path == "/api/delete":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            file_url = body.get("url", "")
            if file_url.startswith("/files/img/") or file_url.startswith("/files/vid/"):
                is_img = file_url.startswith("/files/img/")
                base_dir = OUT_IMG_DIR if is_img else OUT_VID_DIR
                fname = os.path.basename(file_url)
                fpath = base_dir / fname
                if fpath.is_file():
                    fpath.unlink()
                    if STATE.get("last_file") == file_url:
                        STATE["last_file"] = ""
                    return self._send_json({"ok": True})
            self._send_json({"error": "File not found"}, 404)
        else:
            self.send_error(404)


if __name__ == "__main__":
    print(f"[STUDIO WEB] Running on http://192.168.10.168:{WEB_PORT}")
    server = ThreadingHTTPServer((WEB_HOST, WEB_PORT), StudioHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        kill_backend()
        server.server_close()
        http.close()
