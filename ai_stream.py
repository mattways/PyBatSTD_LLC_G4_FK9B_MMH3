import os
import re
import time
import requests
import json
import threading
import torch
import io
import soundfile as sf
from f5_tts.api import F5TTS
from obswebsocket import obsws, requests as obs_req
from pygame import mixer
from flask import Flask, request, render_template_string

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cuda.enable_mem_efficient_sdp(True)
device = torch.device("cuda")

OBS_HOST, OBS_PORT, OBS_PASSWORD, OBS_SCENE = "localhost", 4455, "", "Scene"
CHARACTER_OBS_SOURCES = {"zhorik": "GIF_Zhorik", "vedushiy": "GIF_Vedushiy"}
CHARACTER_ITEM_IDS = {"zhorik": None, "vedushiy": None}

mixer.init()
app = Flask(__name__)

CURRENT_TOPIC = "Срочный экстренный эфир Сев-Кав ТВ"
TOPIC_LOCK = threading.Lock()

HTML_PAGE = """
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Сев-Кав ТВ</title>
<style>body{font-family:sans-serif;background:#121214;color:#fff;text-align:center;padding:40px 20px;}
.card{background:#1a1a1e;max-width:500px;margin:0 auto;padding:30px;border-radius:16px;border:1px solid #2d2d35;}
h1{color:#ff4757;}input[type="text"]{width:100%;padding:14px;background:#25252b;border:2px solid #3d3d48;border-radius:8px;color:#fff;box-sizing:border-box;}
button{background:#ff4757;color:#fff;border:none;padding:14px;border-radius:8px;cursor:pointer;width:100%;font-weight:bold;margin-top:15px;}</style></head>
<body><div class="card"><h1>Сев-Кав ТВ 🎥</h1><form method="POST" action="/"><input type="text" name="topic" placeholder="Какой бред вбросим?..." required autocomplete="off"><button type="submit">ВБРОСИТЬ⚡</button></form></div></body></html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    global CURRENT_TOPIC
    if request.method == "POST":
        t_text = request.form.get("topic", "").strip()
        if t_text:
            with TOPIC_LOCK:
                CURRENT_TOPIC = t_text
    return render_template_string(HTML_PAGE)

threading.Thread(target=lambda: app.run(host="localhost", port=5000, threaded=True), daemon=True).start()

ws = obsws(host=OBS_HOST, port=OBS_PORT, password=OBS_PASSWORD)
ws.connect()
for char_id, source_name in CHARACTER_OBS_SOURCES.items():
    CHARACTER_ITEM_IDS[char_id] = ws.call(obs_req.GetSceneItemId(sceneName=OBS_SCENE, sourceName=source_name)).getSceneItemId()

# Чистая инициализация F5TTS без ломающих try/except и exit(1)
f5tts = F5TTS()

def load_ref(char_id):
    p = f"characters/{char_id}/sample.txt"
    return open(p, "r", encoding="utf-8").read().strip() if os.path.exists(p) else ""

REF_TEXTS = {"zhorik": load_ref("zhorik"), "vedushiy": load_ref("vedushiy")}

def toggle_obs(character_id, enabled):
    item_id = CHARACTER_ITEM_IDS.get(character_id)
    if item_id is not None:
        ws.call(obs_req.SetSceneItemEnabled(sceneName=OBS_SCENE, sceneItemId=item_id, sceneItemEnabled=enabled))

def generate_voice_f5_fast(text, character_id):
    aud = f"characters/{character_id}/sample.wav"
    r_txt = REF_TEXTS.get(character_id, "")
    if not os.path.exists(aud) or not r_txt: return None
    with torch.no_grad():
        res_audio, sample_rate, _ = f5tts.infer(ref_file=aud, ref_text=r_txt, gen_text=text, nfe_step=16)
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, res_audio.cpu().numpy() if torch.is_tensor(res_audio) else res_audio, sample_rate, format='WAV')
    audio_buffer.seek(0)
    torch.cuda.empty_cache()
    return audio_buffer

def clean_gemma_tags(text):
    return re.sub(r'<think>.*?</think>|<\|think\|>|.*?<\/think>|<\|channel\|>thought.*?<channel\|>', '', text, flags=re.DOTALL).strip()

def stream_sentences_from_gemma(character_id, chat_history, current_topic):
    if character_id == "zhorik":
        role = (
            f"Ты — Жорик Вартанов, легендарный ведущий Сев-Кав ТВ (Пятигорск). Твой стиль: уличный кавказский колорит, крики, слова 'Уася', 'жи есть'. "
            f"СЕЙЧАС В ЭФИР ПРИШЕЛ ВБРОС: '{current_topic}'. Твоя задача: взять эту фразу, раскрутить из нее полнейший, абсурдный кавказский бред "
            f"и агрессивно доказывать свою безумную правоту гостю. Если начинаешь дико орать, СРАЗУ ГОВОРИ: "
            f"'Тысяча извинений, Эдуард Аристархович, мы же в прямом эфире, жи есть!' и продолжай спор."
        )
    else:
        role = (
            f"Ты — Доктор Наук (Эдуард Аристархович), интеллигентный, но смертельно взбешенный профессор СКФУ. Твой стиль: заумный, академический язык ('деструктивный дискурс'). "
            f"Ты считаешь Жорика абсолютным дикарем. ТВОЕ ГЛАВНОЕ ПРАВИЛО: Ты обязан жестко БЫКОВАТЬ на Жорика, максимально изысканно, высокомерно "
            f"и агрессивно оскорблять его интеллект за тот бред, который он сочиняет вокруг темы '{current_topic}'! "
            f"Ставь его на место жесткими фразами типа: 'Вы, Жора, продукт интеллектуальной стагнации! Закройте свой рот!'."
        )
    
    sys_p = f"{role} Отвечай ОЧЕНЬ коротко (2 предложения), агрессивно, харизматично. Пиши иностранные названия русскими буквами. ЗАПРЕЩЕНО думать и выдавать теги мыслей в финальный ответ! КРИТИЧЕСКОЕ ПРАВИЛО: Тебе ЗАПРЕЩЕНО соглашаться, поддакивать или повторять слова и мысли оппонента! Переворачивай его слова против него самого!"
    messages = [{"role": "system", "content": sys_p}]
    for speaker, text in chat_history[-4:]:
        messages.append({"role": "assistant" if speaker == character_id else "user", "content": text})
        
    try:
        res = requests.post(
            "http://localhost:8080/v1/chat/completions", 
            json={"messages": messages, "max_tokens": 140, "stream": True}, 
            stream=True, 
            timeout=10
        )
        if res.status_code != 200:
            yield "Связь барахлит, жи есть!"
            return

        buffer = ""
        sentence_endings = re.compile(r'([^.!?]*[.!?])')
        for line in res.iter_lines():
            if not line: continue
            line_str = line.decode('utf-8').strip()
            if line_str.startswith("data:"):
                data_content = line_str[5:].strip()
                if data_content == "[DONE]": break
                try:
                    chunk = json.loads(data_content)
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        buffer += delta['content']
                        while True:
                            match = sentence_endings.match(buffer)
                            if not match: break
                            ready_part = match.group(1).strip()
                            buffer = buffer[match.end():]
                            clean_part = clean_gemma_tags(ready_part)
                            if len(clean_part) > 5: yield clean_part
                except: pass
        final_part = clean_gemma_tags(buffer)
        if len(final_part) > 5: yield final_part
    except:
        yield "Э, ты чё несешь, жи есть!"

def execute_step(character_id, real_name, chat_history):
    with TOPIC_LOCK:
        topic = CURRENT_TOPIC
    print(f"\n🎙️ [{real_name}] (Вброс: {topic}) начинает генерацию...", end="", flush=True)
    full_reply = []
    for sentence in stream_sentences_from_gemma(character_id, chat_history, topic):
        if not sentence: continue
        print(f"\n  ↳ Сгенерирован чанк: \"{sentence}\" -> Синтез аудио...")
        full_reply.append(sentence)
        audio_stream = generate_voice_f5_fast(sentence, character_id)
        if audio_stream:
            toggle_obs(character_id, True)
            mixer.music.load(audio_stream, "wav")
            mixer.music.play()
            while mixer.music.get_busy(): time.sleep(0.07)
            toggle_obs(character_id, False)
    return " ".join(full_reply)

try:
    generate_voice_f5_fast("Тест", "zhorik")
    hist = [("vedushiy", f"Эфир Сев-Кав ТВ начался. Жорик, ну и какую ахинею ты собираешься нести сегодня?")]
    
    while True:
        rep = execute_step("zhorik", "Жорик Вартанов", hist)
        if rep.strip(): 
            hist.append(("zhorik", rep))
            try:
                with open("stream_log.txt", "a", encoding="utf-8") as log_f:
                    log_f.write(f"[{time.strftime('%H:%M:%S')}] ЖОРИК: {rep}\n")
            except: pass
        time.sleep(0.3)
        
        rep = execute_step("vedushiy", "Ведущий Сев-Кав ТВ", hist)
        if rep.strip(): 
            hist.append(("vedushiy", rep))
            try:
                with open("stream_log.txt", "a", encoding="utf-8") as log_f:
                    log_f.write(f"[{time.strftime('%H:%M:%S')}] ДОКТОР: {rep}\n")
            except: pass
        
        if len(hist) > 4: hist = hist[-2:]
        time.sleep(0.7)
except KeyboardInterrupt: pass
finally:
    try: ws.disconnect()
    except: pass
