import os
import re
import time
import requests
import random
import threading
import torch
import io
import soundfile as sf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
device = torch.device("cuda")

from f5_tts.api import F5TTS
from obswebsocket import obsws, requests as obs_req
from pygame import mixer
from flask import Flask, request, render_template_string

OBS_HOST, OBS_PORT, OBS_PASSWORD, OBS_SCENE = "10.9.9.3", 4455, "omHJZMbXA4fT1OwH", "Scene"
CHARACTER_OBS_SOURCES = {"zhorik": "GIF_Zhorik", "vedushiy": "GIF_Vedushiy"}
CHARACTER_ITEM_IDS = {"zhorik": None, "vedushiy": None}
TOPIC_FILE = "topic.txt"
AUTO_TOPICS = [
    "Цены на бензин и огромные пробки в Пятигорске",
    "Кто посмел украсть золотую цепочку у Рудика",
    "Why заниженная Приора в сто раз лучше любого Мерседеса",
    "Тонировка в круг — это по закону или чисто по кайфу?",
    "Куда пропала самая лучшая шаурма на всем Кавказе",
    "Почему Жорик опять умудрился опоздать на прямой эфир",
    "Прогноз погоды: почему в Пятигорске всегда дикая жара, жи есть",
    "Новый Айфон за миллион рублей — стоит ли брать кредит на сто лет"
]

if not os.path.exists(TOPIC_FILE):
    with open(TOPIC_FILE, "w", encoding="utf-8") as f: f.write(AUTO_TOPICS[0])

mixer.init()
app = Flask(__name__)
web_new_topic = None

HTML_PAGE = """
<!DOCTYPE html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Сев-Кав ТВ</title>
<style>body{font-family:sans-serif;background:#121214;color:#fff;text-align:center;padding:40px 20px;}
.card{background:#1a1a1e;max-width:500px;margin:0 auto;padding:30px;border-radius:16px;border:1px solid #2d2d35;}
h1{color:#ff4757;}input[type="text"]{width:100%;padding:14px;background:#25252b;border:2px solid #3d3d48;border-radius:8px;color:#fff;box-sizing:border-box;}
button{background:#ff4757;color:#fff;border:none;padding:14px;border-radius:8px;cursor:pointer;width:100%;font-weight:bold;margin-top:15px;}</style></head>
<body><div class="card"><h1>Сев-Кав ТВ 🎥</h1><form method="POST" action="/"><input type="text" name="topic" placeholder="Новая тема..." required autocomplete="off"><button type="submit">ВБРОСИТЬ⚡</button></form></div></body></html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    global web_new_topic
    if request.method == "POST":
        t_text = request.form.get("topic", "").strip()
        if t_text: web_new_topic = t_text
    return render_template_string(HTML_PAGE)

def run_web_server():
    import logging
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    app.run(host="127.0.0.1", port=5000, threaded=True)

threading.Thread(target=run_web_server, daemon=True).start()

try:
    ws = obsws(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
    ws.connect()
except: exit(1)

for char_id, source_name in CHARACTER_OBS_SOURCES.items():
    try: CHARACTER_ITEM_IDS[char_id] = ws.call(obs_req.GetSceneItemId(sceneName=OBS_SCENE, sourceName=source_name)).getSceneItemId()
    except: pass

try:
    f5tts = F5TTS()
    if hasattr(f5tts, 'model'): f5tts.model = f5tts.model.to(device)
except: ws.disconnect(); exit(1)

def load_reference_text(char_id):
    txt_path = f"characters/{char_id}/sample.txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f: return f.read().strip()
    return ""

REF_TEXTS = {"zhorik": load_reference_text("zhorik"), "vedushiy": load_reference_text("vedushiy")}

def check_for_new_topic(current_topic, start_time, duration=100):
    global web_new_topic
    if web_new_topic is not None:
        sel = web_new_topic; web_new_topic = None
        try:
            with open(TOPIC_FILE, "w", encoding="utf-8") as f: f.write(sel)
        except: pass
        return sel, time.time(), True
    try:
        if os.path.exists(TOPIC_FILE):
            with open(TOPIC_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content and content != current_topic: return content, time.time(), True
    except: pass
    if time.time() - start_time > duration:
        new_t = random.choice([t for t in AUTO_TOPICS if t != current_topic])
        try:
            with open(TOPIC_FILE, "w", encoding="utf-8") as f: f.write(new_t)
        except: pass
        return new_t, time.time(), True
    return current_topic, start_time, False

def toggle_obs_source(character_id, is_speaking):
    item_id = CHARACTER_ITEM_IDS.get(character_id)
    if item_id is not None:
        try: ws.call(obs_req.SetSceneItemEnabled(sceneName=OBS_SCENE, sceneItemId=item_id, sceneItemEnabled=is_speaking))
        except: pass

def generate_gemma_text(character_id, chat_history, current_topic, is_new_topic=False):
    instr = f"Тема изменилась на: '{current_topic}'. Срочно начни дикий ор!" if is_new_topic else f"Спор по теме: '{current_topic}'."
    if character_id == "zhorik":
        role = "Ты Жорик Вартанов телеведущий телеканала постоянно врешь в эфире пытаешься говорить неправду и споришь с опонентом гостем в эфире ты очень злой самодостаточный кавсказский мужчина и против каких либо запретов обожаешь жить по кайфу матерись часто не жестко но сразу же говори тысячи извинений мы в эфире после этого"
    else:
        role = "Ты Доктор Наук гость в эфире телеканала Ты взбешен и НЕНАВИДИШЬ ведущего Жорика! Оскорбляй его за тупость, называй бараном, угрожай разбить лицо в эфире или пожаловаться в ментовку. Требуй закрыть рот и вести себя вежлево."
    
    sys_p = f"{role} Отвечай ОЧЕНЬ коротко (2 предложения), агрессивно, харизматично. Пиши иностранные названия русскими буквами. ЗАПРЕЩЕНО думать и выдавать теги мыслей в финальный ответ! КРИТИЧЕСКОЕ ПРАВИЛО: Тебе ЗАПРЕЩЕНО соглашаться, поддакивать или повторять слова и мысли оппонента! Переворачивай его слова против него самого! {instr}"
    
    messages = [{"role": "system", "content": sys_p}]
    for speaker, text in chat_history[-5:]:
        role_type = "assistant" if speaker == character_id else "user"
        messages.append({"role": role_type, "content": text})
        
    try:
        res = requests.post(
            "http://localhost:8080/v1/chat/completions", 
            json={
                "messages": messages, 
                "temperature": 1.0, 
                "top_p": 0.95,
                "top_k": 64,
                "max_tokens": 64,
                "stream": False
            }, 
            timeout=15
        )
        if res.status_code != 200: return "Слушай, связь барахлит, жи есть!"
        raw = res.json()['choices'][0]['message']['content']
        
        # Точная очистка специфичных токенов рассуждения Gemma 4 E2B
        clean = re.sub(r'<\|channel\|>thought.*?<channel\|>|<\|channel>thought.*?<channel\|>|<think>.*?</think>|<\|think\|>', '', raw, flags=re.DOTALL)
        return clean.replace("<||channel>thought", "").replace("<|channel>thought", "").replace("<channel|>", "").replace('\\n', '\n').replace('\\"', '"').strip()
    except: return "Э, ты чё несешь, жи есть!"

def generate_voice_f5_fast(text, character_id):
    aud = f"characters/{character_id}/sample.wav"
    r_txt = REF_TEXTS.get(character_id, "")
    if not os.path.exists(aud) or not r_txt: return None
    with torch.no_grad():
        res_audio, sample_rate, _ = f5tts.infer(ref_file=aud, ref_text=r_txt, gen_text=text, nfe_step=6)
    audio_np = res_audio.cpu().numpy() if torch.is_tensor(res_audio) else res_audio
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_np, sample_rate, format='WAV')
    audio_buffer.seek(0)
    torch.cuda.empty_cache()
    return audio_buffer

def execute_step(character_id, real_name, chat_history, topic, is_new):
    text = generate_gemma_text(character_id, chat_history, topic, is_new)
    print(f"\n💬 [{real_name}]: {text}")
    audio_stream = generate_voice_f5_fast(text, character_id)
    if audio_stream:
        toggle_obs_source(character_id, True)
        mixer.music.load(audio_stream, "wav")
        mixer.music.play()
        while mixer.music.get_busy(): time.sleep(0.05)
        toggle_obs_source(character_id, False)
    return text

try:
    try: generate_voice_f5_fast("Тест", "zhorik")
    except: pass
    act_topic = "Цены на бензин и пробки в городе"
    if os.path.exists(TOPIC_FILE):
        with open(TOPIC_FILE, "r", encoding="utf-8") as f:
            c = f.read().strip()
            if c: act_topic = c
    t_start = time.time()
    hist = [("vedushiy", f"Эфир Сев-Кав ТВ начался. Жорик, у нас новая тема: {act_topic}. Чё ты скажешь на это?")]
    while True:
        act_topic, t_start, sw = check_for_new_topic(act_topic, t_start)
        rep = execute_step("zhorik", "Жорик Вартанов", hist, act_topic, sw)
        hist.append(("zhorik", rep))
        time.sleep(0.5)
        
        act_topic, t_start, sw = check_for_new_topic(act_topic, t_start)
        rep = execute_step("vedushiy", "Ведущий Сев-Кав ТВ", hist, act_topic, sw)
        hist.append(("vedushiy", rep))
        if len(hist) > 8: hist = hist[-6:]
        time.sleep(0.5)
except KeyboardInterrupt: pass
finally:
    try: ws.disconnect()
    except: pass
