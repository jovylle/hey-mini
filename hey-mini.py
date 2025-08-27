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
# ---------- Config ----------
load_dotenv()
ACCESS_KEY = os.getenv("PV_ACCESS_KEY") or ""
KEYWORD_PATH = os.getenv("PV_KEYWORD_PATH") or ""
SAMPLE_RATE = 16000
CHANNELS = 1
VAD_MODE = 2
MAX_SPEECH_SEC = 20
SILENCE_TAIL_MS = 800
START_WAV = "sounds/start.wav"
HIT_WAV = "sounds/hotword.wav"
PROC_WAV = "sounds/processing.wav"

client = OpenAI()

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
    
    # STT = WhisperModel("tiny.en", compute_type="int8")
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
                beep(HIT_WAV); say("Yes?")
                beep(START_WAV)
                with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as rec:
                    audio_bytes = record_until_silence(vad, rec)
                beep(PROC_WAV)
                text = transcribe(audio_bytes); print("You said:", text)
                
                if not text:
                  say("Sorry, I didn't catch that.")
                  continue
                result = nlu(text)

                if result["action"] == "set_alarm":
                    dt = dateparser.parse(result["time_text"] or text, settings={"PREFER_DATES_FROM":"future"})
                    if dt:
                        make_reminder(dt, text)
                        say(result["reply"] or f"Alarm set for {dt.strftime('%I:%M %p on %B %d')}.")
                    else:
                        say("I couldn't parse the time.")
                elif result["action"] == "get_news":
                    topic = (result.get("topic") or "").strip()
                    region = (result.get("region") or "PH").strip()
                    try:
                        headlines = fetch_news(topic=topic, region=region, limit=3)
                        if headlines:
                            say(result["reply"] or "Here are top stories.")
                            for h in headlines:
                                say(h)
                        else:
                            say("No headlines found.")
                    except Exception:
                        say("News is unavailable right now.")
                else:
                    say(result.get("reply", "Okay."))

if __name__ == "__main__":
    try: main_loop()
    except KeyboardInterrupt: sys.exit(0)
