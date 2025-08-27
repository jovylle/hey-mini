import os, queue, sys, time, subprocess, re, wave
import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel
from dotenv import load_dotenv
import dateparser
import pvporcupine
from openai import OpenAI
import feedparser, urllib.parse

import soundfile as sf
import logging


# ---------- Config ----------
load_dotenv()
ACCESS_KEY = os.getenv("PV_ACCESS_KEY") or ""
KEYWORD_PATH = os.getenv("PV_KEYWORD_PATH") or ""
SAMPLE_RATE = 16000
CHANNELS = 1
VAD_MODE = 2
MAX_SPEECH_SEC = 20
SILENCE_TAIL_MS = 600
START_WAV = "sounds/start.wav"
HIT_WAV = "sounds/hotword.wav"
PROC_WAV = "sounds/mixkit-message-pop-alert-2354.wav"

client = OpenAI()

sd.default.latency = (0.0, 0.0)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("/tmp/hey-mini.log"), logging.StreamHandler()]
)


def load_wav_np(path):
    data, sr = sf.read(path, dtype="float32")
    if sr != 16000:  # optional resample skip if already 16k
        pass
    return data, sr

HIT_SND, HIT_SR = load_wav_np(HIT_WAV)
START_SND, START_SR = load_wav_np(START_WAV)
PROC_SND, PROC_SR = load_wav_np(PROC_WAV)

def beep_np(data, sr):
    sd.play(data, sr, blocking=False)
    
    

def beep(path):
    subprocess.run(["afplay", path], check=False)

def to_pcm16(float_frames):
    return (np.clip(float_frames, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

def write_wav(path, pcm_bytes, sample_rate=16000):
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

def record_until_silence(vad, sd_stream):
    frame_ms = 30
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    voiced, silence, audio = 0, 0, bytearray()
    start = time.time()
    while True:
        data, _ = sd_stream.read(frame_len)
        pcm = to_pcm16(data)
        audio += pcm
        if vad.is_speech(pcm, SAMPLE_RATE): silence = 0; voiced += frame_ms
        else: silence += frame_ms
        if silence >= SILENCE_TAIL_MS and voiced > 0: break
        if time.time() - start > MAX_SPEECH_SEC: break
    return bytes(audio)

def transcribe(pcm_bytes):
    tmp = "/tmp/hey-mini.wav"; write_wav(tmp, pcm_bytes)
    segments, _ = STT.transcribe(tmp, vad_filter=True, language="en")
    return " ".join([seg.text.strip() for seg in segments]).strip()

def parse_alarm(text):
    if not re.search(r"\b(alarm|remind)\b", text, re.I): return None
    return dateparser.parse(text, settings={"PREFER_DATES_FROM":"future"})

def make_reminder(dt, text):
    title = f"Alarm: {text}"
    ds = dt.strftime("%m/%d/%Y %H:%M:%S")
    script = f'''
    tell application "Reminders"
      set newReminder to make new reminder at end of list "Reminders" with properties {{name:"{title}"}}
      set remind me date of newReminder to date "{ds}"
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=False)

def say(text): subprocess.run(["say", text], check=False)

def nlu(text):
    prompt = f"""You are a tiny intent parser named blueberry. Return JSON only:
{{
  "action": one of ["set_alarm","get_news","echo"],
  "time_text": string,  // when action=set_alarm else ""
  "topic": string,      // when action=get_news; e.g., "technology", "sports", "Philippines"
  "region": string,     // ISO country or plain text like "PH" or "Philippines"; empty if unknown
  "reply": string       // short confirmation (<=12 words)
}}

User: {text}"""
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt, temperature=0)
    out = resp.output_text
    import json, re
    m = re.search(r'\{.*\}', out, re.S)
    return json.loads(m.group(0)) if m else {"action":"echo","time_text":"","topic":"","region":"","reply":text}



def fetch_news(topic="", region="PH", limit=3):
    # Google News RSS (region defaults to Philippines, English)
    hl = "en-PH" if (region or "").upper().startswith("PH") else "en-US"
    gl = "PH" if (region or "").upper().startswith("PH") else "US"
    ceid = f"{gl}:{hl.split('-')[0]}"
    if topic:
        q = urllib.parse.quote(topic)
        url = f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
    else:
        url = f"https://news.google.com/rss?hl={hl}&gl={gl}&ceid={ceid}"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:limit]:
        title = e.title.strip()
        source = getattr(e, "source", {}).get("title", "")
        items.append(title if not source else f"{title} — {source}")
    return items
    
def main_loop():
    global STT
    
    logging.info("Booting hey-mini…")
    logging.info("Whisper model: base.en; VAD_MODE=%s; SILENCE_TAIL_MS=%s", VAD_MODE, SILENCE_TAIL_MS)

    STT = WhisperModel("base.en", compute_type="int8")  
    vad = webrtcvad.Vad(VAD_MODE)
    porcupine = pvporcupine.create(access_key=ACCESS_KEY, keywords=["blueberry"])
    frame_length = porcupine.frame_length; sr = porcupine.sample_rate
    q = queue.Queue()
    def cb(indata, frames, t, status): q.put(indata.copy())
    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", blocksize=frame_length, callback=cb):
        print("Listening for 'blue berry'…")
        while True:
            indata = q.get()
            pcm16 = np.frombuffer(to_pcm16(indata), dtype=np.int16)
            if porcupine.process(pcm16) >= 0:
                # 1) Hotword hit → quick ping
                logging.info("Wake word detected.")
                beep_np(HIT_SND, HIT_SR)

                # 2) Start listening cue
                
                logging.info("Start listening.")
                beep_np(START_SND, START_SR)

                # 3) Record until silence (no TTS here to avoid mic bleed)
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as rec:
                    audio_bytes = record_until_silence(vad, rec)
                    
                logging.info("Recording finished: %d bytes", len(audio_bytes))

                # 4) Processing cue (user can stop talking; expect answer)
                beep_np(PROC_SND, PROC_SR)
                logging.info("Transcribing…")

                # 5) Transcribe + act
                text = transcribe(audio_bytes); print("You said:", text)
                logging.info("Transcript: %s", text if text else "<empty>")
                
                if not text:
                    logging.info("No speech; replying apology.")
                    say("Sorry, I didn't catch that.")
                    continue

                try:
                    result = nlu(text)
                    logging.info("NLU: %s", result)

                except Exception:
                    logging.exception("NLU error")
                    say("I had an error understanding that.")
                    continue

                if result["action"] == "set_alarm":
                    logging.info("Action: set_alarm; time_text=%s", result.get("time_text"))

                    dt = dateparser.parse(result.get("time_text") or text,
                                        settings={"PREFER_DATES_FROM":"future"})
                    if dt:
                        make_reminder(dt, text)
                        say(result.get("reply") or f"Alarm set for {dt.strftime('%I:%M %p on %B %d')}.")
                    else:
                        say("I couldn't parse the time.")
                elif result["action"] == "get_news":
                    logging.info("Action: get_news; topic=%s region=%s", result.get("topic"), result.get("region"))
                    topic = (result.get("topic") or "").strip()
                    region = (result.get("region") or "PH").strip()
                    try:
                        headlines = fetch_news(topic=topic, region=region, limit=3)
                        if headlines:
                            say(result.get("reply") or "Here are top stories.")
                            for h in headlines:
                                say(h)
                        else:
                            say("No headlines found.")
                    except Exception:
                        say("News is unavailable right now.")
                else:
                    logging.info("Action: echo")
                    logging.info("Echoing user input: %s", text)
                    say(result.get("reply", "Okay."))

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception:
        logging.exception("Fatal error")
        sys.exit(1)