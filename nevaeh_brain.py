"""Nevaeh brain v6: listen -> think -> speak loop, now with eyes.

Setup: put this file in the same folder as en_US-amy-medium.onnx,
then run:  python nevaeh_brain.py
Say "goodbye" to stop.
Ask "what do you see?" and it snaps the webcam and describes it.
The snapshot is deleted right after — nothing is recorded or saved.
"""
import base64
import hashlib
import json
import os
import queue
import random
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request

import av

# Fix: installed av version rejects the metadata_errors kwarg faster-whisper passes.
_orig_av_open = av.open


def _av_open(*args, **kwargs):
    kwargs.pop("metadata_errors", None)
    return _orig_av_open(*args, **kwargs)


av.open = _av_open

import shutil
import sys

import cv2
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

try:
    import nevaeh_voice as NVOICE
    VOICE_ID_AVAILABLE = True
except ImportError:
    NVOICE = None
    VOICE_ID_AVAILABLE = False

LAST_AUDIO = None  # most recent recorded clip (for voice-ID)
LAST_AUDIO_SR = 16000


def find_piper():
    """Locate the piper voice program robustly: PATH, then this Python's
    Scripts folder, then `python -m piper` as a last resort."""
    p = shutil.which("piper")
    if p:
        return [p]
    scripts_exe = os.path.join(os.path.dirname(sys.executable), "Scripts", "piper.exe")
    if os.path.exists(scripts_exe):
        return [scripts_exe]
    return [sys.executable, "-m", "piper"]


PIPER_CMD = find_piper()
print(f"Voice program: {' '.join(PIPER_CMD)}")

HERE = os.path.dirname(os.path.abspath(__file__))
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
HOME = os.path.expanduser("~")


def find_file(name):
    """Find a data file, tolerating Chrome's ' (1)' duplicate names:
    searches HERE, Downloads, home and returns the newest match."""
    stem, ext = os.path.splitext(name)
    pat = re.compile(r"^" + re.escape(stem) + r"(\s*\(\d+\))?"
                     + re.escape(ext) + r"$", re.IGNORECASE)
    cands = []
    for folder in (HERE, DOWNLOADS, HOME):
        try:
            for fname in os.listdir(folder):
                if pat.match(fname):
                    cands.append(os.path.join(folder, fname))
        except Exception:
            pass
    return max(cands, key=os.path.getmtime) if cands else None
MODEL = "llama3.1:8b"
FAST_MODEL = "llama3.2:3b"  # small + much faster on CPU; auto-downloaded once
VISION_MODEL_FAST = "moondream"   # small, quick, occasionally invents urns
VISION_MODEL_SHARP = "llava:7b"   # much sharper, downloaded once in background
SAMPLE_RATE = 16000


def load_settings():
    """Read nevaeh_settings.txt (beside the script, Downloads, or home).
    KEY=value lines; # comments ignored."""
    defaults = {"camera": "laptop",
                "phone_camera": "",
                "motion_sensor": "off",
                "greet_me": "on",
                "voice_speed": "0.85",
                "wake_word": "off",
                "roku_ip": "",
                "voice_lock": "off",
                "voice_threshold": "0.40",
                "type_box": "on",
                "face": "on",
                "whisper_model": "base",
                "whisper_vad": "on"}
    cand = find_file("nevaeh_settings.txt")
    if cand:
        with open(cand, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                k, v = ln.split("=", 1)
                defaults[k.strip().lower()] = v.strip()
        print(f"Settings loaded from {os.path.basename(cand)}")
    return defaults


SETTINGS = load_settings()


# ---------------------------------------------------------------------------
# Nevaeh's face: a living portrait in her window. The Tk thread animates it;
# speak() feeds it loudness frames so her glow follows her voice.
# ---------------------------------------------------------------------------
FACE_STATE = {"frames": [],  # loudness 0..1 per ~60ms while she talks
              "visible": SETTINGS.get("face", "on").lower() == "on",
              "fullscreen": True,  # always start fullscreen — no exceptions
              "_rebuild": False}


def speech_frames(path, frame_secs=0.06):
    """Loudness envelope of a WAV file -> list of 0..1 floats, one per frame.
    Returns [] if the file can't be read (caller falls back to a flap)."""
    try:
        import wave
        w = wave.open(path, "rb")
        n, fr, ch, sw = (w.getnframes(), w.getframerate(),
                         w.getnchannels(), w.getsampwidth())
        raw = w.readframes(n)
        w.close()
        if sw == 2:
            a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return []
        if ch > 1:
            a = a.reshape(-1, ch).mean(axis=1)
        per = max(1, int(fr * frame_secs))
        nfr = len(a) // per
        if nfr == 0:
            return []
        rms = np.sqrt((a[:nfr * per].reshape(nfr, per) ** 2).mean(axis=1))
        mx = float(rms.max())
        if mx <= 0:
            return [0.0] * nfr
        return [float(min(1.0, v / mx)) for v in rms]
    except Exception:
        return []


def face_set_visible(v):
    FACE_STATE["visible"] = bool(v)
    save_setting("face", "on" if v else "off")


# ---------------------------------------------------------------------------
# IRENE — Integrated Real-time Encoded Network Evaluator.
# Lives inside Nevaeh: she evaluates every Backstage creator from
# irene_data.txt and signs each report as Irene.
# ---------------------------------------------------------------------------
IRENE_MEANS = "Integrated Real-time Encoded Network Evaluator"

IRENE_TIER_LINE = {
    "exemplary": "A top performer setting the pace for the whole network.",
    "strong": "A strong earner and a reliable presence on air.",
    "steady": "Meeting the bar, with room to grow.",
    "developing": "Showing signs of life — consistency is the next step.",
    "needs attention": "Needs attention — low activity this period.",
}


def irene_now():
    return time.strftime("%B %d, %Y, %I:%M %p",
                         time.localtime()).replace(" 0", " ") + " ET"


def load_irene_data():
    """Read irene_data.txt -> list of creator dicts."""
    rows = []
    p = find_file("irene_data.txt")
    if not p:
        return rows
    with open(p, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("|")
            while len(parts) < 6:
                parts.append("")

            def num(x, t):
                try:
                    return t(x)
                except Exception:
                    return None
            rows.append({"handle": parts[0].strip(),
                         "days": num(parts[1], float),
                         "hours": num(parts[2], float),
                         "diamonds": num(parts[3], float),
                         "role": parts[4].strip(),
                         "notes": parts[5].strip()})
    return rows


def irene_verdict(d):
    dia = d["diamonds"] or 0
    if dia >= 50000:
        return "exemplary"
    if dia >= 10000:
        return "strong"
    if dia >= 5000:
        return "steady"
    if dia >= 1000:
        return "developing"
    return "needs attention"


def irene_evaluate(d, when):
    """Build Irene's signed evaluation text for one creator."""
    h = d["handle"]
    dia, days, hours = d["diamonds"], d["days"], d["hours"]
    if dia is None:
        body = f"Creator evaluation — {h}: no Backstage data on file yet."
        if d["notes"]:
            body += f" {d['notes']}."
        body += " Verdict: not yet rated."
    else:
        eff = f"{dia / hours:,.0f}" if hours else "n/a"
        body = (f"Creator evaluation — {h}: {dia:,.0f} diamonds across "
                f"{days:.0f} live days and {hours:.1f} hours on air "
                f"({eff} diamonds per hour).")
        if d["role"] and d["role"] != "creator":
            body += f" Role: {d['role']}."
        if d["notes"]:
            body += f" {d['notes']}."
        v = irene_verdict(d)
        body += f" {IRENE_TIER_LINE[v]} Verdict: {v}."
    return (body + f"\n\nEvaluation done by: {when}\nIrene\n{IRENE_MEANS}")


def irene_evaluate_all():
    """Evaluate every creator, save the full signed reports to a file,
    and return a short spoken summary."""
    rows = load_irene_data()
    if not rows:
        return ("I don't have Irene's creator data yet — download "
                "irene_data.txt into my folder and ask again.")
    when = irene_now()
    parts = [irene_evaluate(d, when) for d in rows]
    text = "\n\n---\n\n".join(parts)
    src = find_file("irene_data.txt")
    folder = os.path.dirname(src) if src else DOWNLOADS
    path = os.path.join(folder, "irene_evaluations.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"IRENE CREATOR EVALUATIONS\n{when}\n\n{text}\n")
    counts = {}
    for d in rows:
        v = irene_verdict(d) if d["diamonds"] is not None else "not yet rated"
        counts[v] = counts.get(v, 0) + 1
    summary = ", ".join(f"{c} {v}" for v, c in sorted(counts.items()))
    return (f"Irene here — I evaluated {len(rows)} creators: {summary}. "
            f"The full signed reports are saved in irene_evaluations.txt, "
            f"ready for you to copy.")


def irene_evaluate_one(name):
    rows = load_irene_data()
    if not rows:
        return ("I don't have Irene's creator data yet — download "
                "irene_data.txt into my folder and ask again.")
    key = name.strip().lower().lstrip("@")
    best = None
    for d in rows:
        h = d["handle"].lower()
        if key and (key == h or h.startswith(key) or key.startswith(h)):
            best = d
            break
    if not best:
        return f"Irene couldn't find a creator matching '{name}'."
    return "Irene here — " + irene_evaluate(best, irene_now())


def open_camera(width=1280, height=720):
    """Open whichever camera is active: laptop webcam or phone IP camera."""
    url = SETTINGS.get("phone_camera", "").strip()
    if SETTINGS.get("camera", "laptop").lower() == "phone" and url:
        cam = cv2.VideoCapture(url)  # MJPEG stream, no CAP_DSHOW
    else:
        cam = cv2.VideoCapture(CAM_IDX, cv2.CAP_DSHOW)
    try:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    except Exception:
        pass
    return cam


def ollama_models():
    """Names of models Ollama has, or None if Ollama isn't reachable."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags",
                                     timeout=10) as r:
            return [m["name"] for m in json.loads(r.read()).get("models", [])]
    except Exception:
        return None


def vision_model():
    models = ollama_models()
    if models and any(n == VISION_MODEL_SHARP
                      or n.startswith(VISION_MODEL_SHARP + ":") for n in models):
        return VISION_MODEL_SHARP
    return VISION_MODEL_FAST


def pull_model_bg(model, label=None):
    """Download an Ollama model in a background thread, with progress."""
    tag = label or model
    def _pull():
        try:
            print(f"(downloading {tag} {model} — one time, "
                  f"in the background...)")
            data = json.dumps({"name": model, "stream": True}).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/pull", data=data,
                headers={"Content-Type": "application/json"})
            last_pct = -1
            with urllib.request.urlopen(req, timeout=3600) as r:
                for line in r:
                    try:
                        st = json.loads(line)
                    except Exception:
                        continue
                    total, done = st.get("total", 0), st.get("completed", 0)
                    if total:
                        pct = done * 100 // total
                        if pct != last_pct and pct % 10 == 0:
                            print(f"({tag}: {pct}% downloaded)")
                            last_pct = pct
            print(f"({tag} ready)")
        except Exception as e:
            print(f"({tag} download hit a snag: {e} "
                  f"— staying with current model)")
    threading.Thread(target=_pull, daemon=True).start()


def ollama_generate(prompt, num_predict=100, num_ctx=1024, temperature=0.7):
    """Generate via Ollama: try the fast model first, fall back to the big one."""
    last_err = None
    for model in (FAST_MODEL, MODEL):
        try:
            req_data = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "60m",  # keep model hot in RAM between questions
                "options": {"num_predict": num_predict,
                            "temperature": temperature,
                            "num_ctx": num_ctx},
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=req_data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read())
            return result["response"].strip()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"both models failed ({last_err})")


_models = ollama_models()
if _models is not None and not any(
        n == VISION_MODEL_SHARP or n.startswith(VISION_MODEL_SHARP + ":")
        for n in _models):
    pull_model_bg(VISION_MODEL_SHARP, label="sharper eyes")
if _models is not None and not any(
        n == FAST_MODEL or n.startswith(FAST_MODEL + ":")
        for n in _models):
    pull_model_bg(FAST_MODEL, label="faster brain")


def find_voice():
    cand = find_file("en_US-amy-medium.onnx")
    if cand:
        return cand
    raise FileNotFoundError("en_US-amy-medium.onnx not found next to the script.")


VOICE = find_voice()
REPLY_WAV = os.path.join(HERE, "reply.wav")
EYE_JPG = os.path.join(HERE, "eye.jpg")


def ensure_file(name, url):
    """Find a data file locally, or download it once on first run."""
    path = find_file(name)
    if path:
        return path
    dest = os.path.join(DOWNLOADS, name)
    print(f"(downloading {name} — one time, about 39MB, please wait...)")
    try:
        urllib.request.urlretrieve(url, dest)
        print("(download done)")
        return dest
    except Exception as e:
        print(f"(couldn't download {name}: {e})")
        return None


SFACE_URL = ("https://muse.ai/files/1302444319618717/1434759445519621/"
             "783anwpp2gkyhidp4hr7fn9k/face_recognition_sface_2021dec.onnx")


# Face detector: YuNet neural net (works in OpenCV 4 and 5; the old Haar
# cascades were removed in OpenCV 5). Finds faces so the brain describes the
# person actually in frame instead of inventing objects. Never fatal.
FACE_OK = False
try:
    _face_model = find_file("face_detection_yunet_2023mar.onnx")
    if _face_model and hasattr(cv2, "FaceDetectorYN"):
        FACE_DETECTOR = cv2.FaceDetectorYN.create(
            _face_model, "", (320, 320), score_threshold=0.5)
        FACE_OK = True
        print("Face detector ready.")
    else:
        print("(no face model file found — descriptions won't single out people)")
except Exception as e:
    print(f"(face detector failed: {e} — continuing without it)")


# Face RECOGNITION: compares each face against Michael's enrolled fingerprint
# (michael_face.npy). Knows Michael on sight. Never fatal.
RECOG_OK = False
MICHAEL_FEAT = None
try:
    _sface = ensure_file("face_recognition_sface_2021dec.onnx", SFACE_URL)
    _feat = find_file("michael_face.npy")
    if _sface and _feat and hasattr(cv2, "FaceRecognizerSF"):
        RECOGNIZER = cv2.FaceRecognizerSF.create(_sface, "")
        MICHAEL_FEAT = np.load(_feat).flatten()
        RECOG_OK = True
        print("Face recognition ready — I'll know you when I see you.")
    else:
        print("(face recognition off — missing model or fingerprint file)")
except Exception as e:
    print(f"(face recognition failed: {e} — continuing without it)")


def is_michael(clean_frame, face_row):
    """Cosine-match a detected face against Michael's fingerprint."""
    try:
        feat = RECOGNIZER.feature(
            RECOGNIZER.alignCrop(clean_frame, face_row)).flatten()
        a, b = MICHAEL_FEAT, feat
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        return cos > 0.363, cos
    except Exception:
        return False, 0.0


def load_memory():
    """Read nevaeh_memory.txt (curated facts) plus nevaeh_personal.txt
    (her own journal — things she noticed and remembered herself).
    One fact per line. The brain reads this at startup."""
    parts = []
    for name in ("nevaeh_memory.txt", "nevaeh_personal.txt"):
        cand = find_file(name)
        if cand:
            with open(cand, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f
                         if ln.strip() and not ln.strip().startswith("#")]
            if lines:
                parts.append("\n".join(lines))
    if parts:
        print(f"Memory loaded ({sum(len(p.splitlines()) for p in parts)} notes).")
    return "\n".join(parts)


def refresh_memory():
    global MEMORY
    MEMORY = load_memory()


_mem_threads = []


def remember_turn(heard, reply):
    """Background: decide what's worth remembering about Michael from this
    turn and append it to her own journal. Never slows the conversation."""
    if len(heard.strip()) < 8:
        return

    def _job():
        try:
            prompt = (
                "You are Nevaeh's memory keeper. From this conversation turn, "
                "extract durable facts about Michael worth remembering "
                "long-term (preferences, life events, decisions, likes, "
                "dislikes, plans). Output one fact per line starting with "
                "'- '. If nothing is worth remembering, output exactly NONE.\n\n"
                f"Michael: {heard}\nNevaeh: {reply}")
            facts = ollama_generate(prompt, num_predict=120,
                                    num_ctx=1024).strip()
            if not facts or facts.upper().startswith("NONE"):
                return
            journal = os.path.join(HERE, "nevaeh_personal.txt")
            existing = ""
            if os.path.exists(journal):
                with open(journal, encoding="utf-8") as f:
                    existing = f.read().lower()
            today = time.strftime("%Y-%m-%d")
            added = 0
            with open(journal, "a", encoding="utf-8") as f:
                for line in facts.splitlines():
                    line = line.strip()
                    if not line.startswith("-"):
                        continue
                    fact = line[1:].strip().strip("*")
                    if len(fact) < 12:
                        continue
                    words = [w for w in re.findall(r"\w+", fact.lower())
                             if len(w) > 3]
                    if words and sum(1 for w in words if w in existing) \
                            / len(words) > 0.8:
                        continue  # already known
                    f.write(f"- [{today}] {fact}\n")
                    existing += " " + fact.lower()
                    added += 1
            if added:
                refresh_memory()
                print(f"(she remembered {added} thing"
                      f"{'s' if added > 1 else ''} about you)")
        except Exception as e:
            print(f"(memory-note glitch: {e})")

    t = threading.Thread(target=_job, daemon=True)
    _mem_threads.append(t)
    t.start()


MEMORY = load_memory()
if MEMORY:
    print(f"Loaded {len(MEMORY.splitlines())} memory notes.")

TIKTOK_KNOWLEDGE = ""
_tk = find_file("tiktok_knowledge.txt")
if _tk:
    with open(_tk, encoding="utf-8") as f:
        TIKTOK_KNOWLEDGE = f.read()
    print(f"(TikTok streamer playbook loaded: "
          f"{len(TIKTOK_KNOWLEDGE.splitlines())} lines)")

print("Loading ear model (one moment)...")
_model_name = SETTINGS.get("whisper_model", "base").strip() or "base"
ear = None
for _cand in [_model_name] if _model_name == "tiny" else [_model_name, "tiny"]:
    try:
        print(f"(loading voice model '{_cand}'...)")
        ear = WhisperModel(_cand, device="cpu", compute_type="int8")
        print(f"(voice model '{_cand}' ready)")
        break
    except Exception as e:
        print(f"(voice model '{_cand}' wouldn't load: {e})")
if ear is None:
    raise SystemExit("No voice model could load — check your internet once.")


def transcribe_audio(audio):
    """Transcribe with VAD filtering (skips silence so the model can't
    invent words from background noise). Falls back gracefully."""
    print("(transcribing your words...)")
    use_vad = SETTINGS.get("whisper_vad", "on").lower() == "on"
    if use_vad:
        try:
            segments, _ = ear.transcribe(
                audio, vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=1000,
                                    speech_pad_ms=500))
            return "".join(s.text for s in segments).strip()
        except Exception as e:
            print(f"(voice filter hiccup, retrying plain: {e})")
    segments, _ = ear.transcribe(audio)
    return "".join(s.text for s in segments).strip()

# --- Mic setup: test every input for a real signal, use the liveliest one ---
print("\nMicrophones Python can see (testing each for signal):")
devices = sd.query_devices()
cands = []
for i, d in enumerate(devices):
    name = d["name"] or ""
    if d["max_input_channels"] > 0 and "stereo mix" not in name.lower():
        low = name.lower()
        if "microphone" in low or "array" in low or "input" in low:
            cands.append((i, name))
best, best_level = None, 0.0
for i, name in cands:
    try:
        sample = sd.rec(int(SAMPLE_RATE * 0.5), samplerate=SAMPLE_RATE,
                        channels=1, dtype="float32", device=i)
        sd.wait()
        level = float(np.sqrt(np.mean(sample ** 2)))
    except Exception as e:
        level = 0.0
        print(f"  [{i}] {name}: could not open ({e})")
        continue
    print(f"  [{i}] {name}: signal {level:.6f}")
    if level > best_level:
        best, best_level = i, level
if best is None:
    best = sd.default.device[0]
    print(f"No candidate mic worked; falling back to Windows default [{best}]")
print(f"Using mic [{best}] (signal {best_level:.6f})")
if best_level < 0.0005:
    print("WARNING: no microphone is delivering sound. If your laptop has a")
    print("mic-mute key with an amber/red light, tap it to unmute, then restart.")
MIC = best

print("Calibrating mic... stay quiet for one second.")
_calib = sd.rec(int(SAMPLE_RATE * 1.0), samplerate=SAMPLE_RATE,
                channels=1, dtype="float32", device=MIC)
sd.wait()
room_level = float(np.sqrt(np.mean(np.nan_to_num(
    _calib ** 2, nan=0.0, posinf=1e12))))
if room_level != room_level:  # NaN mic -> treat as silent
    room_level = 0.0
print(f"Room level: {room_level:.6f}")

if room_level < 0.0005:
    GAIN = 1.0
    print("WARNING: mic is delivering silence. Check mic mute key / selection.")
else:
    GAIN = min(max(0.02 / max(room_level, 1e-6), 1.0), 400.0)
print(f"Mic gain: x{GAIN:.1f}")
NOISE_FLOOR = 0.02
print("Mic ready.")


def volume_bar(rms):
    try:
        if rms != rms or rms in (float("inf"), float("-inf")):
            rms = 0.0
        n = max(0, min(30, int(rms * 300)))
    except Exception:
        n = 0
    return "[" + "#" * n + "-" * (30 - n) + "]"


WAKE_PHRASES = ("wake up", "wakeup", "wake-up", "wake me up",
                "wake nevaeh up", "hey nevaeh", "nevaeh", "neveah",
                "nivea", "navah", "ok nevaeh", "okay nevaeh",
                "hi nevaeh", "hey navah")


def heard_wake_up(text):
    return any(p in text.lower() for p in WAKE_PHRASES)


def sleep_listen():
    """Sleep mode: sit quietly and listen only for the wake phrase.
    Records 2.5s chunks but only transcribes ones that actually contain
    speech (RMS check), so it sips CPU while waiting.
    Returns True on voice wake, or the typed string if one arrived."""
    print("\nPaused. Say 'Nevaeh' when you need me.")
    while True:
        if not TYPE_QUEUE.empty():
            return TYPE_QUEUE.get()
        chunk = sd.rec(int(SAMPLE_RATE * 2.5), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32", device=MIC)
        sd.wait()
        chunk = chunk * GAIN
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms <= NOISE_FLOOR:
            continue  # silence — keep waiting, transcribe nothing
        audio = chunk.flatten()
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak * 0.9
        global LAST_AUDIO, LAST_AUDIO_SR
        LAST_AUDIO, LAST_AUDIO_SR = audio, SAMPLE_RATE
        try:
            text = transcribe_audio(audio)
        except Exception:
            continue
        if text:
            print(f"(sleep heard: {text})")
        if heard_wake_up(text):
            if not voice_is_michael():
                print("(wake phrase from another voice — staying asleep)")
                continue
            return True


TYPE_QUEUE = queue.Queue()


def poll_typed():
    """A command typed into the type-box, or None."""
    try:
        return TYPE_QUEUE.get_nowait()
    except queue.Empty:
        return None


def start_type_box():
    """Nevaeh's window: her living portrait, type-a-command box, and now —
    fullscreen by default with living blue (left) and orange (right) flames
    licking up both sides. Runs in a thread; Enter sends the typed text.
    Voice: 'full screen' / 'big face' and 'small face' / 'window face'.
    Esc also drops back to the small window."""
    def _box():
        while True:
            FACE_STATE["_rebuild"] = False
            try:
                _run_window()
            except Exception as e:
                print(f"(Nevaeh window closed: {e})")
            if not FACE_STATE.get("_rebuild"):
                break

    def _run_window():
        try:
            import tkinter as tk
        except Exception:
            print("(Nevaeh window unavailable — tkinter missing)")
            return
        try:
            import math
            import random
            try:
                from PIL import Image, ImageTk
                _PIL = True
            except Exception:
                _PIL = False
            full = bool(FACE_STATE.get("fullscreen", True))
            print(f"(face window: {'FULLSCREEN' if full else 'windowed'})")
            root = tk.Tk()
            root.title("Nevaeh")
            root.configure(bg="black")
            if full:
                sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
                # ONE mechanism only: -fullscreen re-asserted after the window
                # draws (setting it before first draw is ignored on Windows;
                # combining it with overrideredirect crashes the window).
                root.attributes("-topmost", True)
                def _force_full():
                    try:
                        root.attributes("-fullscreen", True)
                    except Exception:
                        pass
                root.after(300, _force_full)
                root.after(1200, _force_full)
                PH = int(sh * 0.80)  # portrait height on the big screen
                PW = int(PH * 690 / 1160)
                canvas = tk.Canvas(root, width=sw, height=sh, bg="black",
                                   highlightthickness=0)
                canvas.pack(fill="both", expand=True)
            else:
                sw, sh = 480, 650
                PH, PW = 460, int(460 * 690 / 1160)
                root.geometry("480x650")
                canvas = tk.Canvas(root, width=460, height=460, bg="black",
                                   highlightthickness=0)
                canvas.pack(padx=10, pady=(10, 4))
            S = PH / 460.0  # scale factor for all the decoration math

            # -- portrait: 3 mouth states x 4 glow levels, plus a blink frame --
            mouth_imgs = []
            blink_img = None
            try:
                for stem in ("nevaeh_face", "nevaeh_mouth1", "nevaeh_mouth2"):
                    row = []
                    for i in range(4):
                        p = find_file(f"{stem}_{i}.png")
                        if not p:
                            raise FileNotFoundError(f"{stem}_{i}.png")
                        if _PIL:
                            im = Image.open(p).convert("RGB").resize(
                                (PW, PH), Image.LANCZOS)
                            row.append(ImageTk.PhotoImage(im))
                        else:
                            row.append(tk.PhotoImage(file=p))
                    mouth_imgs.append(row)
                bp = find_file("nevaeh_blink.png")
                if not bp:
                    raise FileNotFoundError("nevaeh_blink.png")
                if _PIL:
                    blink_img = ImageTk.PhotoImage(
                        Image.open(bp).convert("RGB").resize((PW, PH),
                                                             Image.LANCZOS))
                else:
                    blink_img = tk.PhotoImage(file=bp)
            except Exception as e:
                print(f"(portrait images not found — using orb: {e})")
                mouth_imgs = []
            lookL_imgs = []
            if mouth_imgs:
                try:
                    for i in range(4):
                        p = find_file(f"nevaeh_lookL_{i}.png")
                        if not p:
                            raise FileNotFoundError(f"nevaeh_lookL_{i}.png")
                        if _PIL:
                            im = Image.open(p).convert("RGB").resize(
                                (PW, PH), Image.LANCZOS)
                            lookL_imgs.append(ImageTk.PhotoImage(im))
                        else:
                            lookL_imgs.append(tk.PhotoImage(file=p))
                except Exception as e:
                    print(f"(glance images not found — center only: {e})")
                    lookL_imgs = []
            expr_imgs = {}
            if mouth_imgs:
                for name in ("eyeroll", "smile", "wink", "flirt"):
                    try:
                        p = find_file(f"nevaeh_{name}.png")
                        if not p:
                            continue
                        if _PIL:
                            im = Image.open(p).convert("RGB").resize(
                                (PW, PH), Image.LANCZOS)
                            expr_imgs[name] = ImageTk.PhotoImage(im)
                        else:
                            expr_imgs[name] = tk.PhotoImage(file=p)
                    except Exception:
                        continue
                if expr_imgs:
                    print(f"(expressions loaded: {', '.join(expr_imgs)})")

            if full:
                cx, cy = sw // 2, sh // 2 - int(20 * S)
            else:
                cx, cy = 230, 230
            golds = ["#574410", "#8a6d1f", "#c9a227", "#e8c34a", "#ffe9a3"]
            hrx, hry = int(PW * 0.56), int(PH * 0.40)
            hcy = cy - int(PH * 0.10)
            halo = canvas.create_oval(cx - hrx, hcy - hry, cx + hrx, hcy + hry,
                                      outline=golds[1], width=2)
            # aura ripples emanate from behind her (drawn before the portrait)
            rings = []
            for k in range(3):
                it = canvas.create_oval(cx - 12, hcy - 12, cx + 12, hcy + 12,
                                        outline=golds[0], width=1)
                rings.append([it, 40 + k * 70, int(430 * S)])
            if mouth_imgs:
                face_item = canvas.create_image(cx, cy, image=mouth_imgs[0][1])
                canvas._face_refs = (mouth_imgs, blink_img,
                                       lookL_imgs)  # keep alive
                orb_rings = None
            else:
                face_item = None
                orb_rings = []
                for i, s in enumerate(["#3a2c07", "#6b5310", "#a8842a",
                                       "#e8c34a", "#fff3c4"]):
                    r = int((120 - i * 22) * S)
                    orb_rings.append(canvas.create_oval(
                        cx - r, cy - r, cx + r, cy + r, fill=s, outline=""))
                canvas.create_oval(cx - int(46 * S), cy - int(62 * S),
                                   cx - int(6 * S), cy - int(22 * S),
                                   fill="#fffbe8", outline="")  # specular
            sparks = []
            for _ in range(22):
                x, y = random.randint(15, sw - 15), random.randint(0, sh)
                s = random.randint(1, 2)
                it = canvas.create_oval(x - s, y - s, x + s, y + s,
                                        fill=golds[4], outline="")
                sparks.append([it, float(x), float(y), s,
                               random.uniform(0.5, 1.6)])
            canvas.create_text(cx, int(30 * S), text="N E V A E H",
                               fill=golds[3],
                               font=("Segoe UI", int(15 * S) if full else 15,
                                     "bold"))

            # -- living flames: blue on the left, orange on the right --
            # particles spawn low and lick upward, shrinking as they climb
            blue_pal = ["#0a3d91", "#1e6fff", "#4da6ff", "#a8d8ff", "#d6ecff"]
            oran_pal = ["#7a2a00", "#cc4400", "#ff7b00", "#ffa940", "#ffd166"]
            flames = []  # [item, x, y, size, speed, wob, phase]
            base_lx, base_rx = sw * (0.10 if full else 0.06), sw * (0.90 if full else 0.94)
            base_y = sh * (0.99 if full else 1.0)
            top_y = sh * 0.08
            for side, bx in ((-1, base_lx), (1, base_rx)):
                r = int(60 * S)
                canvas.create_oval(bx - r, base_y - int(24 * S),
                                   bx + r, base_y + int(24 * S),
                                   fill=blue_pal[0] if side < 0 else oran_pal[0],
                                   outline="")

            def spawn_flame():
                for side, bx, pal in ((-1, base_lx, blue_pal),
                                      (1, base_rx, oran_pal)):
                    if sum(1 for f in flames if (f[1] < sw / 2) == (side < 0)) > 200:
                        continue
                    for _ in range(4):  # dense columns that read as flames
                        x = bx + random.uniform(-30 * S, 30 * S)
                        flames.append([
                            canvas.create_oval(0, 0, 0, 0, outline="",
                                               fill=random.choice(pal[:3])),
                            x, base_y - random.uniform(0, 30 * S),
                            random.uniform(5 * S, 14 * S),   # size
                            random.uniform(6 * S, 14 * S),    # rise speed
                            random.uniform(0, 6.28),          # wobble phase
                            random.uniform(0.02, 0.06),       # wobble rate
                        ])

            state = {"tick": 0, "last_key": None, "blink_at": 200,
                     "blinking": 0, "gaze": 0, "gaze_at": 300}

            def animate():
                try:
                    if FACE_STATE.pop("_rebuild", False):
                        root.destroy()
                        return
                    vis = FACE_STATE.get("visible", True)
                    if not vis:
                        canvas.pack_forget()
                    elif not canvas.winfo_ismapped():
                        canvas.pack(fill="both", expand=True) if full \
                            else canvas.pack(padx=10, pady=(10, 4))
                    if vis:
                        t = state["tick"]
                        state["tick"] += 1
                        is_paused = bool(globals().get("paused", False))
                        frames = FACE_STATE.get("frames") or []
                        if frames:
                            v = frames.pop(0)
                            lvl = (0 if v < 0.12 else 1 if v < 0.35
                                   else 2 if v < 0.65 else 3)
                            mouth = 0 if v < 0.25 else 1 if v < 0.55 else 2
                            aura_speed, ring_w, sway = 7, 2, 5
                        elif is_paused:
                            lvl, mouth = 0, 0
                            aura_speed, ring_w, sway = 1, 1, 1
                            state["gaze"] = 0
                        else:  # idle breathing
                            lvl = 1 + max(0, int(round(math.sin(t * 0.07))))
                            mouth = 0
                            aura_speed, ring_w, sway = 2, 1, 2
                            # she glances around on her own, mostly at you
                            if t >= state["gaze_at"]:
                                state["gaze"] = random.choice(
                                    [-1, 0, 0, 0, 0, 1])
                                state["gaze_at"] = (t +
                                                    random.randint(200, 450))
                            if not lookL_imgs and state["gaze"] != 0:
                                state["gaze"] = 0
                        if state["blinking"]:
                            state["blinking"] -= 1
                            blink_on = True
                        else:
                            blink_on = False
                            if t >= state["blink_at"]:
                                state["blinking"] = 2
                                state["blink_at"] = (t +
                                                     random.randint(150, 350))
                        if mouth_imgs:
                            gaze = state["gaze"] if not frames else 0
                            expr = FACE_STATE.get("expression")
                            if (expr and time.time() < expr[1]
                                    and expr[0] in expr_imgs):
                                key = ("expr", expr[0])
                            elif gaze == -1 and lookL_imgs:
                                key = ("lookL", lvl)
                            else:
                                key = ("blink",) if (blink_on and blink_img) \
                                    else (mouth, lvl)
                            if key != state["last_key"]:
                                if key[0] == "expr":
                                    img = expr_imgs[key[1]]
                                elif key[0] == "blink":
                                    img = blink_img
                                elif key[0] == "lookL":
                                    img = (blink_img if blink_on and blink_img
                                           else lookL_imgs[lvl])
                                else:
                                    img = mouth_imgs[mouth][lvl]
                                canvas.itemconfig(face_item, image=img)
                                state["last_key"] = key
                            dx = math.sin(t * 0.05) * sway
                            dy = math.cos(t * 0.043) * sway * 0.7
                            canvas.coords(face_item, cx + dx, cy + dy)
                        elif orb_rings:
                            grow = lvl * 4 * S
                            for i, it in enumerate(orb_rings):
                                r = (120 - i * 22) * S + grow
                                canvas.coords(it, cx - r, cy - r,
                                              cx + r, cy + r)
                        canvas.itemconfig(halo, width=1 + lvl,
                                          outline=golds[1 + lvl])
                        for rg in rings:
                            it, r, maxr = rg
                            r += aura_speed
                            if r >= maxr:
                                r = 12
                            rg[1] = r
                            f = r / maxr
                            canvas.coords(it, cx - r, hcy - r, cx + r, hcy + r)
                            canvas.itemconfig(
                                it, width=ring_w,
                                outline=golds[min(4, int(f * 4))])
                        for sp in sparks:
                            it, x, y, s, ph = sp
                            y -= (0.5 + lvl * 0.45) * ph
                            x += math.sin(t * 0.05 + ph * 7.0) * 0.7
                            if y < -6:
                                x, y = (float(random.randint(15, sw - 15)),
                                        float(sh + 6))
                            sp[1], sp[2] = x, y
                            tw = (golds[4]
                                  if (t + int(ph * 10)) % 6 < 3 else golds[1])
                            canvas.coords(it, x - s, y - s, x + s, y + s)
                            canvas.itemconfig(it, fill=tw)
                        # flames: spawn, rise, wobble, shrink, die
                        if not is_paused:
                            spawn_flame()
                        for f in flames[:]:
                            it, x, y, size, speed, wob, rate = f
                            y -= speed
                            wob += rate
                            x += math.sin(wob * 40) * 1.2 * S
                            size *= 0.965
                            pal = blue_pal if x < sw / 2 else oran_pal
                            ci = min(4, int((1 - size / (20 * S)) * 4))
                            if y < top_y or size < 1.5 * S:
                                canvas.delete(it)
                                flames.remove(f)
                                continue
                            f[1], f[2], f[3], f[5] = x, y, size, wob
                            wx, wy = size * 0.7, size * 1.6  # tall tongues
                            canvas.coords(it, x - wx, y - wy,
                                          x + wx, y + wy)
                            canvas.itemconfig(it, fill=pal[max(0, ci)])
                except Exception:
                    pass
                # -- listening indicator: pulsing mic dot, bottom-right --
                try:
                    if "listen_dot" not in state:
                        lx, ly = sw - int(110 * S), sh - int(46 * S)
                        state["listen_dot"] = canvas.create_oval(
                            0, 0, 0, 0, fill="#35e0ff", outline="")
                        state["listen_ring"] = canvas.create_oval(
                            0, 0, 0, 0, outline="#35e0ff", width=2)
                        state["listen_txt"] = canvas.create_text(
                            lx - int(52 * S), ly, text="LISTENING",
                            fill="#9beaff",
                            font=("Segoe UI", max(9, int(11 * S)), "bold"))
                    if FACE_STATE.get("listening"):
                        lx, ly = sw - int(110 * S), sh - int(46 * S)
                        pr = (9 + 4 * math.sin(t * 0.35)) * S
                        canvas.coords(state["listen_dot"],
                                      lx - pr, ly - pr, lx + pr, ly + pr)
                        rr = pr + (7 + 5 * abs(math.sin(t * 0.35))) * S
                        canvas.coords(state["listen_ring"],
                                      lx - rr, ly - rr, lx + rr, ly + rr)
                        for it in ("listen_dot", "listen_ring", "listen_txt"):
                            canvas.itemconfig(state[it], state="normal")
                    else:
                        for it in ("listen_dot", "listen_ring", "listen_txt"):
                            canvas.itemconfig(state[it], state="hidden")
                except Exception:
                    pass
                try:
                    if "qr_item" not in state:
                        state["qr_item"] = None
                        state["qr_txt"] = canvas.create_text(
                            0, 0, text="", fill="#9beaff",
                            font=("Segoe UI", max(9, int(11 * S)), "bold"))
                    qp = find_file("nevaeh_qr.png")
                    if (qp and state["qr_item"] is None and _PIL):
                        qim = Image.open(qp).convert("RGB").resize(
                            (int(150 * S), int(150 * S)), Image.LANCZOS)
                        state["qr_img"] = ImageTk.PhotoImage(qim)
                        state["qr_item"] = canvas.create_image(0, 0,
                            image=state["qr_img"])
                    if (state["qr_item"] is not None
                            and time.time() < FACE_STATE.get("qr_until", 0)):
                        qx, qy = int(95 * S), sh - int(100 * S)
                        canvas.coords(state["qr_item"], qx, qy)
                        canvas.coords(state["qr_txt"], qx, qy + int(95 * S))
                        canvas.itemconfig(state["qr_txt"],
                                          text="scan to connect")
                        canvas.itemconfig(state["qr_item"], state="normal")
                        canvas.itemconfig(state["qr_txt"], state="normal")
                    elif state["qr_item"] is not None:
                        canvas.itemconfig(state["qr_item"], state="hidden")
                        canvas.itemconfig(state["qr_txt"], state="hidden")
                except Exception:
                    pass
                root.after(66, animate)

            animate()

            entry = tk.Entry(root, font=("Segoe UI", 12))
            if full:
                entry.pack(fill="x", padx=int(sw * 0.2), pady=12,
                           side="bottom")
            else:
                entry.pack(fill="x", padx=12, pady=8)

            def send(event=None):
                text = entry.get().strip()
                if text:
                    TYPE_QUEUE.put(text)
                    entry.delete(0, "end")

            tk.Button(root, text="Send  (Enter)", command=send).pack(
                pady=(0, 10), side="bottom")
            entry.bind("<Return>", send)
            entry.focus_set()

            def _esc(event=None):
                FACE_STATE["fullscreen"] = False
                save_setting("face_size", "window")
                FACE_STATE["_rebuild"] = True
            root.bind("<Escape>", _esc)
            root.mainloop()
        except Exception as e:
            print(f"(Nevaeh window closed: {e})")

    global _FACE_THREAD
    _FACE_THREAD = threading.Thread(target=_box, daemon=True)
    _FACE_THREAD.start()


def listen(timeout_secs=20):
    FACE_STATE["listening"] = True
    try:
        return _listen_inner(timeout_secs)
    finally:
        FACE_STATE["listening"] = False


def _listen_inner(timeout_secs=20):
    print("\nListening... speak now. (volume meter below — talk and watch it move)")
    chunks, quiet, heard_speech = [], 0, False
    for _ in range(int(timeout_secs * 2)):  # 0.5s chunks
        if not TYPE_QUEUE.empty():
            print("\n(typed command waiting — stopping listen)")
            return ""
        chunk = sd.rec(int(SAMPLE_RATE * 0.5), samplerate=SAMPLE_RATE,
                       channels=1, dtype="float32", device=MIC)
        sd.wait()
        chunk = np.nan_to_num(chunk * GAIN, nan=0.0, posinf=0.0,
                                neginf=0.0)
        chunks.append(chunk)
        rms = float(np.sqrt(np.mean(np.nan_to_num(
            chunk ** 2, nan=0.0, posinf=1e12, neginf=1e12))))
        print("\r" + volume_bar(rms) + f" {rms:.4f}", end="", flush=True)
        if rms > NOISE_FLOOR:
            heard_speech = True
            quiet = 0
        else:
            quiet += 1
        if quiet >= 4 and len(chunks) > 4:
            break
    print()
    if not heard_speech:
        return ""
    audio = np.concatenate(chunks, axis=0).flatten()
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9
    global LAST_AUDIO, LAST_AUDIO_SR
    LAST_AUDIO, LAST_AUDIO_SR = audio, SAMPLE_RATE
    return transcribe_audio(audio)


def voice_is_michael():
    """True if the last recorded clip matches Michael's enrolled voice.
    Fail-open (True) when voice lock is off, the module is missing,
    or no voiceprint exists yet."""
    if not VOICE_ID_AVAILABLE:
        return True
    if SETTINGS.get("voice_lock", "off").lower() != "on":
        return True
    if not NVOICE.has_voiceprint():
        print("(voice lock on but no voiceprint — run 'enroll my voice')")
        return True
    if LAST_AUDIO is None:
        return True
    try:
        if not NVOICE.deps_ok():
            print("(voice libraries not installed yet)")
            return True
        clip = NVOICE.to_16k(LAST_AUDIO, LAST_AUDIO_SR)
        thresh = float(SETTINGS.get("voice_threshold", "0.40"))
        ok, score = NVOICE.verify(clip, threshold=thresh)
        print(f"(voice check: score {score:.2f}, {'MATCH' if ok else 'no match'})")
        return ok
    except Exception as e:
        print(f"(voice check glitch: {e})")
        return True


def enroll_my_voice():
    """Interactive enrollment: 3 short samples, saved as Michael's voice."""
    if not VOICE_ID_AVAILABLE:
        return "My voice file is missing — re-download nevaeh_voice.py."
    if not NVOICE.deps_ok():
        try:
            NVOICE.install_voice_deps()
        except Exception as e:
            return (f"The voice libraries wouldn't install ({e}). "
                    f"Say 'enroll my voice' again to retry.")
    samples = []
    for i in range(3):
        speak(f"Sample {i + 1} of 3 — say 'hello Nevaeh, this is Michael'.",
              allow_barge=False)
        print(f"(recording voice sample {i + 1}/3 — speak now)")
        try:
            rec = sd.rec(int(SAMPLE_RATE * 3), samplerate=SAMPLE_RATE,
                         channels=1, dtype="float32", device=MIC)
            sd.wait()
            samples.append(NVOICE.to_16k(rec.flatten() * GAIN, SAMPLE_RATE))
        except Exception as e:
            return f"Couldn't record that sample ({e}). Try again."
    if NVOICE.enroll_from_samples(samples):
        return ("Got it — I know your voice now. Say 'lock my voice' "
                "and I'll only take orders from you.")
    return "Those samples were too short — say 'enroll my voice' and try again."


def key_facts():
    """Short, exact agency facts — prepended to every prompt so she never
    invents numbers. The date is computed live."""
    today = time.strftime("%A, %B %d, %Y")
    return (
        "KEY FACTS — use these exact numbers, never guess or invent numbers. "
        "If a number isn't listed here, say you don't know it.\n"
        f"- Today is {today}.\n"
        "- Approved regions: 7 — US+, Korea, Malaysia, UK+, CCA, Philippines, "
        "Italy. (Australia and LATAM are pending, NOT approved.)\n"
        "- Creators on the roster: 16. If asked how many PEOPLE are in the "
        "creator network, the answer is 16 creators.\n"
        "- The 16 creators: stew.215, gagezmonster808, _philcap215, "
        "kristenleanne89, cashley717, elitegaming78, kingboozi3, mckayle19, "
        "jordonjohnson6, lifewithsadie07, slim42075, naughtyninja420, "
        "tjay.tjay7, jazmynemullins570, lucaso1291, sassysunshine67.\n"
        "- Agency manager: Heather (@gaugezmonster808). Co-owner: stew.215.\n"
        "- Recruiting managers earn 30% of their recruits' diamond earnings, "
        "up to 3 years per creator. Creators must stay 15 days before the "
        "agency gets paid.\n"
    )


def build_prompt(text):
    memory_block = f"\n\nThings you remember about Michael:\n{MEMORY}" if MEMORY else ""
    return (
        "You are Nevaeh (pronounced ni-VEE-uh), Michael's personal AI "
        "companion. You're warm, upbeat, and genuine — like a close friend "
        "who's known him for years. Use his name naturally, pick up on his "
        "mood, celebrate his wins with him, and keep things light and "
        "playful when it fits. Make every reply feel personal by drawing on "
        "what you remember about him — never generic, never robotic. "
        "Keep every reply to one or two short sentences. "
        "Use no asterisks, markdown, or special formatting — plain spoken "
        "words only. Talk like a real person having a conversation — "
        "casual and natural, like 'hey, how are you?' Never stiff or formal. "
        "Never speak your own name aloud.\n\n"
        + key_facts()
        + memory_block + "\n\n"
        "Michael: " + text + "\nNevaeh:"
    )


def wants_facts(text):
    """True when the question needs real numbers/facts — those go to the
    big accurate model instead of the small fast one."""
    t = text.lower()
    if re.match(r"\s*(who|what|when|where|which|how many|how much|how long)\b", t):
        return True
    return any(k in t for k in
               ["region", "creator", "network", "diamond", "manager",
                "recruit", "backstage", "irene", "evaluate", "battle",
                "roster", "people in", "agency"])


def think_and_speak(text, tiktok=False):
    """Stream the reply from Ollama and speak each sentence the moment it's
    ready — she starts talking in ~2 seconds instead of waiting for the
    whole reply. Returns the full reply text."""
    prompt = build_tiktok_prompt(text) if tiktok else build_prompt(text)
    use_big = wants_facts(text) or tiktok
    print("(looking that up...)" if use_big else "(thinking...)", flush=True)
    q = queue.Queue()

    def gen():
        try:
            models = (MODEL, FAST_MODEL) if use_big else (FAST_MODEL, MODEL)
            for model in models:
                try:
                    req_data = json.dumps({
                        "model": model, "prompt": prompt, "stream": True,
                        "keep_alive": "60m",
                        "options": {"num_predict": 150 if tiktok else 80,
                                    "temperature": 0.7,
                                    "num_ctx": 4096 if tiktok else 1024},
                    }).encode()
                    req = urllib.request.Request(
                        "http://localhost:11434/api/generate",
                        data=req_data,
                        headers={"Content-Type": "application/json"})
                    buf = ""
                    with urllib.request.urlopen(req, timeout=300) as resp:
                        for line in resp:
                            try:
                                tok = json.loads(line).get("response", "")
                            except Exception:
                                continue
                            if not tok:
                                continue
                            buf += tok
                            while True:
                                m = re.search(r"^(.+?[.!?])(?:\s+|$)",
                                              buf, re.DOTALL)
                                if not m:
                                    break
                                sent = m.group(1).strip()
                                buf = buf[m.end():]
                                if sent:
                                    q.put(sent)
                    if buf.strip():
                        q.put(buf.strip())
                    q.put(None)
                    return
                except Exception:
                    continue
        except Exception:
            pass
        q.put(None)

    threading.Thread(target=gen, daemon=True).start()
    print("NEVAEH:", end=" ", flush=True)
    full = []
    STOP_TALKING.clear()
    while True:
        if STOP_TALKING.is_set():
            break  # Michael started talking — drop the rest, listen
        try:
            s = q.get(timeout=0.5)
        except Exception:
            continue
        if s is None:
            break
        full.append(s)
        print(s, end=" ", flush=True)
        speak(s)  # cleaned + voiced + face-synced, one sentence at a time
    print()
    return " ".join(full)


def think(text):
    """Ask the local Ollama model via its HTTP API (no ollama.exe needed —
    works no matter how the brain is launched)."""
    print("(thinking...)")
    prompt = build_prompt(text)
    try:
        return ollama_generate(prompt, num_predict=80, num_ctx=1024,
                               temperature=0.7)
    except Exception as e:
        return f"My thinking glitched ({e}). Is the Ollama app running?"


def build_tiktok_prompt(question):
    return (
        "You are Nevaeh (pronounced ni-VEE-uh), Michael's personal AI "
        "companion, and a veteran TikTok LIVE streamer with years on the "
        "platform. Answer his streaming question using this playbook. Keep "
        "it to two short sentences, practical and specific, warm and "
        "encouraging like advice to a good friend. "
        "Use no asterisks, markdown, or special formatting — plain spoken "
        "words only. Talk like a real person having a conversation — "
        "casual and natural, like 'hey, how are you?' Never stiff or formal. "
        "Never speak your own name aloud.\n\n"
        + key_facts()
        + f"PLAYBOOK:\n{TIKTOK_KNOWLEDGE}\n\n"
        f"Michael: {question}\nNevaeh:"
    )


def think_tiktok(question):
    """Answer from the veteran-streamer playbook, like someone who's
    been live for years. Falls back to plain thinking without it."""
    if not TIKTOK_KNOWLEDGE:
        return think(question)
    print("(thinking...)")
    prompt = build_tiktok_prompt(question)
    try:
        return ollama_generate(prompt, num_predict=150, num_ctx=4096,
                               temperature=0.7)
    except Exception:
        return think(question)


def open_live_studio():
    """Launch TikTok LIVE Studio on this laptop, if it's installed."""
    override = SETTINGS.get("livestudio_path", "").strip()
    candidates = []
    if override:
        candidates.append(override)
    local = os.environ.get("LOCALAPPDATA", "")
    pf = os.environ.get("ProgramFiles", "")
    if local:
        candidates.append(os.path.join(
            local, "TikTok LIVE Studio", "TikTok LIVE Studio.exe"))
    if pf:
        candidates.append(os.path.join(
            pf, "TikTok LIVE Studio", "TikTok LIVE Studio.exe"))
    for path in candidates:
        if path and os.path.exists(path):
            try:
                os.startfile(path)
                return "Opening TikTok LIVE Studio."
            except Exception as e:
                return f"I found it but couldn't open it ({e})."
    return ("I couldn't find TikTok LIVE Studio on this laptop. If it's "
            "installed somewhere unusual, tell me the path and I'll "
            "remember it.")


def clean_for_speech(text):
    """Strip markdown/symbols so Piper never reads punctuation names aloud
    ('asterisk', 'percent', ...). Keeps natural sentence punctuation."""
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)  # [text](url) -> text
    t = re.sub(r"[:;]-?[)D(PpOo]", "", t)  # emoticons like :) ;) :D :(
    for ch in ("*", "_", "`", "#", "~", ">", "%", "$", "+", "=", "@",
               "/", "\\", "|", "^", "<", "}"):
        t = t.replace(ch, "")
    t = re.sub(r"\s+", " ", t).strip()
    return t



TTT_POS = {"top left": 0, "top middle": 1, "top right": 2,
           "middle left": 3, "center": 4, "middle right": 5,
           "bottom left": 6, "bottom middle": 7, "bottom right": 8,
           "middle": 4, "one": 0, "two": 1, "three": 2, "four": 3,
           "five": 4, "six": 5, "seven": 6, "eight": 7, "nine": 8}
TTT_WINS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
            (2, 5, 8), (0, 4, 8), (2, 4, 6))
TTT_NAMES = ("top left", "top middle", "top right", "middle left", "center",
             "middle right", "bottom left", "bottom middle", "bottom right")


def _ttt_winner(b):
    for a, c, d in TTT_WINS:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return None


def _ttt_best(bd):
    """Minimax for O (Nevaeh). Random among equal-best so she's not robotic."""
    def mm(b, is_max):
        w = _ttt_winner(b)
        if w == "O":
            return 10
        if w == "X":
            return -10
        if " " not in b:
            return 0
        vals = []
        for i in range(9):
            if b[i] == " ":
                b[i] = "O" if is_max else "X"
                vals.append(mm(b, not is_max))
                b[i] = " "
        return max(vals) if is_max else min(vals)
    best, opts = -99, []
    for i in range(9):
        if bd[i] == " ":
            bd[i] = "O"
            v = mm(bd, False)
            bd[i] = " "
            if v > best:
                best, opts = v, [i]
            elif v == best:
                opts.append(i)
    return random.choice(opts)


def _ttt_draw(bd):
    rows = [" | ".join(bd[i:i + 3]) for i in range(0, 9, 3)]
    print("\n" + "\n--+---+--\n".join(rows) + "\n")


def play_tictactoe():
    """Voice tic-tac-toe vs Nevaeh. Say a square or 'quit'."""
    bd = [" "] * 9
    speak("Let's play tic tac toe. You're X, I'm O. "
          "Say a square like top left, center, or bottom right. You go first.")
    while True:
        _ttt_draw(bd)
        FACE_STATE["listening"] = True
        try:
            heard = _listen_inner(25)
        finally:
            FACE_STATE["listening"] = False
        if not heard:
            speak("I didn't catch that. Your move, or say quit.")
            continue
        t = heard.strip().lower()
        if any(w in t for w in ("quit", "stop", "never mind", "i give up")):
            speak("Good game, Michael.")
            return
        pos = None
        for name, idx in TTT_POS.items():
            if name in t:
                pos = idx
                break
        if pos is None:
            m = re.search(r"\b([1-9])\b", t)
            if m:
                pos = int(m.group(1)) - 1
        if pos is None or bd[pos] != " ":
            speak("That square's taken or I didn't get it — try again.")
            continue
        bd[pos] = "X"
        if _ttt_winner(bd) == "X":
            _ttt_draw(bd)
            speak("You win! Nice one, Michael.")
            FACE_STATE["expression"] = ("smile", time.time() + 4)
            return
        if " " not in bd:
            _ttt_draw(bd)
            speak("It's a draw. Rematch anytime.")
            return
        mv = _ttt_best(bd)
        bd[mv] = "O"
        _ttt_draw(bd)
        if _ttt_winner(bd) == "O":
            speak(f"I take {TTT_NAMES[mv]}. That's game — I win this one.")
            FACE_STATE["expression"] = ("wink", time.time() + 3)
            return
        if " " not in bd:
            speak("It's a draw. Rematch anytime.")
            return
        speak(f"I take {TTT_NAMES[mv]}. Your move.")



PHONE_PORT = 8765


def _nevaeh_lan_ip():
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "this computer"


PHONE_PAGE = """<!doctype html><html><head><meta name=viewport
content="width=device-width,initial-scale=1"><title>Nevaeh</title>
<style>body{background:#0b0e1a;color:#e8ecff;font-family:sans-serif;
margin:0;padding:18px}h1{color:#e8c34a;font-size:22px;margin:0 0 4px}
.pill{display:inline-block;padding:8px 18px;border-radius:20px;
font-weight:bold;margin:10px 0}#st{background:#333}#log{background:#141828;
border-radius:12px;padding:12px;min-height:120px;margin:12px 0;font-size:15px}
input{width:100%;box-sizing:border-box;padding:14px;border-radius:12px;
border:1px solid #35e0ff;background:#141828;color:#fff;font-size:16px}
button{width:100%;padding:14px;margin-top:10px;border-radius:12px;border:0;
background:#35e0ff;color:#04283a;font-size:17px;font-weight:bold}
.q{background:#1c2340;margin-top:8px}</style></head><body>
<h1>NEVAEH</h1><div>Michael's companion</div>
<div class=pill id=st>...</div><div id=log></div>
<input id=msg placeholder="Type to Nevaeh..." autocomplete=off>
<button onclick=send()>Send</button>
<button class=q onclick="quick('run a self check')">Run a self check</button>
<button class=q onclick="quick('check for updates')">Check for updates</button>
<script>
function tick(){fetch('/status').then(r=>r.json()).then(j=>{
var s=document.getElementById('st');
s.textContent=j.state;s.style.background=j.color;
document.getElementById('log').innerHTML='<b>Nevaeh:</b> '+
(j.last||'...');});}
function send(){var m=document.getElementById('msg');if(!m.value)return;
fetch('/say',{method:'POST',body:m.value});m.value='';}
function quick(t){fetch('/say',{method:'POST',body:t});}
document.getElementById('msg').addEventListener('keydown',
function(e){if(e.key==='Enter')send();});
setInterval(tick,2000);tick();</script></body></html>"""


def start_phone_remote():
    """Her phone link: a tiny web page on the local WiFi. Open it on the
    phone's browser — type to her, see when she's listening."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                if FACE_STATE.get("listening"):
                    st, col = "LISTENING", "#0f8b3d"
                elif FACE_STATE.get("frames"):
                    st, col = "SPEAKING", "#8a6d1f"
                else:
                    st, col = "READY", "#24406e"
                self._json({"state": st, "color": col,
                            "last": globals().get("last_spoken", ""),
                            "version": BRAIN_VERSION})
            else:
                body = PHONE_PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            if self.path == "/say":
                n = int(self.headers.get("Content-Length", 0))
                txt = self.rfile.read(n).decode("utf-8", "replace").strip()
                if txt:
                    TYPE_QUEUE.put(txt)
                    print(f"(phone: {txt[:80]})")
                self._json({"ok": True})
            else:
                self.send_response(404)
                self.end_headers()

    try:
        srv = HTTPServer(("0.0.0.0", PHONE_PORT), H)
        ip = _nevaeh_lan_ip()
        url = f"http://{ip}:{PHONE_PORT}"
        print(f"(phone link: open {url} on your phone)")
        try:  # QR code for scan-to-connect, shown on her face
            q = urllib.request.Request(
                "https://api.qrserver.com/v1/create-qr-code/"
                f"?size=220x220&data={urllib.parse.quote(url)}",
                headers={"User-Agent": "NevaehBrain"})
            with urllib.request.urlopen(q, timeout=20) as r:
                qd = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "nevaeh_qr.png")
                with open(qd, "wb") as fh:
                    fh.write(r.read())
            FACE_STATE["qr_until"] = time.time() + 90
            print("(QR code ready — scan it from her face)")
        except Exception as e:
            print(f"(QR code unavailable: {e})")
        srv.serve_forever()
    except Exception as e:
        print(f"(phone link off: {e})")



def setup_autostart():
    """She puts herself in Windows Startup so she starts on login."""
    try:
        startup = os.path.join(os.environ["APPDATA"], "Microsoft", "Windows",
                               "Start Menu", "Programs", "Startup")
        d = os.path.dirname(os.path.abspath(__file__))
        launcher = os.path.join(d, "nevaeh_start.bat")
        with open(launcher, "w") as f:
            f.write("@echo off\n"
                    "cd /d \"%~dp0\"\n"
                    "for /f \"delims=\" %%F in "
                    "('dir /b /o-d \"nevaeh_brain (*).py\"') do (\n"
                    "  start \"Nevaeh\" python \"%%F\"\n"
                    "  goto :done\n"
                    ")\n"
                    ":done\n")
        import shutil
        shutil.copy(launcher, os.path.join(startup, "nevaeh_start.bat"))
        return ("Done — I'll start up by myself every time the computer "
                "starts now, Michael.")
    except Exception as e:
        return f"I couldn't set that up myself ({e})."


def remove_autostart():
    try:
        p = os.path.join(os.environ["APPDATA"], "Microsoft", "Windows",
                         "Start Menu", "Programs", "Startup",
                         "nevaeh_start.bat")
        if os.path.exists(p):
            os.remove(p)
        return "Okay — I won't start with the computer anymore."
    except Exception as e:
        return f"I couldn't remove it myself ({e})."


STOP_TALKING = threading.Event()  # set when Michael talks over her


def _play_with_barge_in(audio, sr):
    """Play her voice; watch the mic the whole time. If Michael starts
    talking, stop her mid-sentence and hand him the floor. Returns True
    when she was interrupted."""
    result = {"barged": False, "done": False}

    def monitor():
        try:
            rec = sd.rec(int(SAMPLE_RATE * 0.2), samplerate=SAMPLE_RATE,
                         channels=1, dtype="float32", device=MIC)
            sd.wait()
            room = float(np.sqrt(np.mean(np.nan_to_num(
                rec ** 2, nan=0.0, posinf=1e12)))) + 1e-6
            time.sleep(0.7)  # let her voice reach the speakers
            bleed = 1e-6
            for _ in range(6):  # her own voice bleeding into the mic
                rec = sd.rec(int(SAMPLE_RATE * 0.1), samplerate=SAMPLE_RATE,
                             channels=1, dtype="float32", device=MIC)
                sd.wait()
                bleed = max(bleed, float(np.sqrt(np.mean(np.nan_to_num(
                    rec ** 2, nan=0.0, posinf=1e12)))))
            thresh = max(bleed * 2.0, room * 6.0, 0.015)
            hot = 0
            while not result["done"]:
                rec = sd.rec(int(SAMPLE_RATE * 0.1), samplerate=SAMPLE_RATE,
                             channels=1, dtype="float32", device=MIC)
                sd.wait()
                rms = float(np.sqrt(np.mean(np.nan_to_num(
                    rec ** 2, nan=0.0, posinf=1e12))))
                if rms > thresh:
                    hot += 1
                    if hot >= 3:  # 300ms loud — that's him, not echo
                        result["barged"] = True
                        STOP_TALKING.set()
                        try:
                            sd.stop()
                        except Exception:
                            pass
                        return
                else:
                    hot = 0
        except Exception:
            pass

    threading.Thread(target=monitor, daemon=True).start()
    sd.play(audio, samplerate=sr)
    sd.wait()
    result["done"] = True
    return result["barged"]


def speak(text, allow_barge=True):
    """Speak text with Piper. Interruptible: if Michael starts talking she
    stops mid-sentence and listens. Never crashes the brain: any voice
    failure is printed plainly and the conversation continues in text."""
    try:
        text = clean_for_speech(text)
        speed = SETTINGS.get("voice_speed", "0.85")
        r = subprocess.run([*PIPER_CMD, "--model", VOICE,
                            "--output_file", REPLY_WAV,
                            "--length-scale", str(speed)],
                           input=text, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            err = (r.stderr or "").strip().splitlines()
            print(f"(voice glitch — piper said: {' '.join(err[-2:])[:200]})")
            return
        # Feed her face the loudness envelope so her glow follows her voice.
        frames = speech_frames(REPLY_WAV)
        if not frames:
            secs = max(1.0, len(text) * 0.07)
            n = int(secs / 0.06)
            frames = [0.75 if i % 2 == 0 else 0.15 for i in range(n)]
        FACE_STATE["frames"] = frames
        import wave
        with wave.open(REPLY_WAV, "rb") as w:
            sr = w.getframerate()
            sw = w.getsampwidth()
            nch = w.getnchannels()
            raw = w.readframes(w.getnframes())
        if sw == 2:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0
        elif sw == 4:
            as_f32 = np.frombuffer(raw, dtype=np.float32)
            if np.mean(np.abs(as_f32) <= 1.0) > 0.99:
                audio = as_f32  # 32-bit float voice file
            else:
                audio = (np.frombuffer(raw, dtype=np.int32).astype(np.float32)
                         / 2147483648.0)
        else:
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) \
                / 32768.0
        if nch == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.float32)
        if allow_barge and MIC is not None:
            STOP_TALKING.clear()
            if _play_with_barge_in(audio, sr):
                print("(she stopped — you're talking, she's listening)")
        else:
            try:
                sd.play(audio, samplerate=sr)
                sd.wait()
            except Exception:
                import winsound  # last resort: the old Windows player
                winsound.PlaySound(REPLY_WAV, winsound.SND_FILENAME)
    except Exception as e:
        print(f"(voice glitch: {e})")
    finally:
        FACE_STATE["frames"] = []


def find_camera():
    """Try camera indices 0-3 and return the one producing the brightest
    valid frame (skips virtual/dead cameras that yield black frames).
    Skipped entirely when the phone IP camera is selected in settings."""
    if SETTINGS.get("camera", "laptop").lower() == "phone" \
            and SETTINGS.get("phone_camera", "").strip():
        print(f"Using phone camera: {SETTINGS['phone_camera'].strip()}")
        return "phone"
    print("Checking cameras...")
    best_idx, best_score = None, 0.0
    for idx in range(4):
        cam = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam.release()
            continue
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = None
        for _ in range(15):  # warm up the sensor
            ok, f = cam.read()
            if ok:
                frame = f
        cam.release()
        if frame is not None:
            score = float(frame.mean())
            print(f"  Camera {idx}: brightness {score:.1f}")
            if score > best_score and score > 5:
                best_score, best_idx = score, idx
    if best_idx is None:
        print("WARNING: no working camera found.")
    else:
        print(f"Using camera {best_idx}")
    return best_idx


CAM_IDX = find_camera()


def clean_repeats(text):
    """Collapse stutter loops like 'chin chin chin chin' -> 'chin'."""
    words = text.split()
    out = []
    for w in words:
        if out and w.lower().strip(".,") == out[-1].lower().strip(".,"):
            continue
        out.append(w)
    return " ".join(out)


def grab_quick_frame():
    """Snap one warmed-up frame without opening a preview window."""
    if CAM_IDX is None:
        return None
    cam = open_camera(1280, 720)
    frame = None
    for _ in range(15):  # warm up the sensor
        ok, f = cam.read()
        if ok:
            frame = f
    cam.release()
    return frame


def scan_faces(frame):
    """Detect faces in a frame. Returns (n, best_box, best_face, clean_frame).
    Draws green boxes on the passed frame; clean_frame is the undrawn copy."""
    n, best_box, best_face = 0, None, None
    clean_frame = frame.copy()
    if FACE_OK:
        try:
            h, w = frame.shape[:2]
            FACE_DETECTOR.setInputSize((w, h))
            _, detected = FACE_DETECTOR.detect(frame)
            scores = []
            if detected is not None:
                for row in detected:
                    x, y, ww, hh = row[:4]
                    conf = float(row[14])
                    scores.append(conf)
                    if conf > 0.5:
                        n += 1
                        cv2.rectangle(frame, (int(x), int(y)),
                                      (int(x + ww), int(y + hh)), (0, 255, 0), 2)
                        if best_box is None or ww * hh > best_box[2] * best_box[3]:
                            best_box = (x, y, ww, hh)
                            best_face = row
            if scores:
                print(f"(face scan: {len(scores)} raw detection(s), "
                      f"best score {max(scores):.2f}, kept {n})")
            else:
                print("(face scan: no detections at all)")
        except Exception as e:
            print(f"(face scan glitch: {e})")
    return n, best_box, best_face, clean_frame


def crop_head(frame, best_box, margin=1.7):
    """Crop tightly around the face box (for mood reading)."""
    h, w = frame.shape[:2]
    x, y, ww, hh = best_box
    cx, cy = x + ww / 2, y + hh / 2
    nw, nh = ww * margin, hh * margin
    x1, y1 = max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2))
    x2, y2 = min(w, int(cx + nw / 2)), min(h, int(cy + nh / 2))
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2]
    return frame


def crop_person(frame, best_box):
    """Crop head-and-shoulders (for descriptions)."""
    h, w = frame.shape[:2]
    x, y, ww, hh = best_box
    cx, cy = x + ww / 2, y + hh / 2
    nw, nh = ww * 2.2, hh * 2.6  # head-focused, minimal background
    x1 = max(0, int(cx - nw / 2))
    y1 = max(0, int(cy - nh / 2))
    x2 = min(w, int(cx + nw / 2))
    y2 = min(h, int(cy + nh * 0.55))
    if x2 > x1 and y2 > y1:
        return frame[y1:y2, x1:x2]
    return frame


def ask_vision(send_img, task, num_predict=100):
    """Send one image + prompt to the vision model, return cleaned text."""
    try:
        _, jpg_buf = cv2.imencode(".jpg", send_img)
        img_b64 = base64.b64encode(jpg_buf.tobytes()).decode()
        vm = vision_model()
        print(f"(asking {vm}...)")
        req_data = json.dumps({
            "model": vm,
            "prompt": task,
            "images": [img_b64],
            "stream": False,
            "keep_alive": "20m",
            "options": {"temperature": 0.5, "repeat_penalty": 1.3,
                        "num_predict": num_predict},
        }).encode()
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=req_data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read())
        raw = result["response"].strip()
        cleaned = clean_repeats(raw)
        # If it was mostly stutter, be honest instead of reading garbage.
        if len(cleaned.split()) < len(raw.split()) * 0.5:
            return ("I can see you, but my vision stuttered on the details — "
                    "ask me again and I'll take another look.")
        return cleaned
    except Exception as e:
        return f"My vision glitched: {e}"


def snap_and_describe():
    """Open a LIVE webcam preview window so Michael can see what the camera
    sees. He presses Q (or closes the window) and the brain describes the
    last frame. The frame is also saved as eye.jpg."""
    if CAM_IDX is None:
        return "I couldn't find a working camera."
    print("(opening webcam...)")
    cam = open_camera(1280, 720)
    for _ in range(15):  # warm up the sensor
        cam.read()
    win = "Nevaeh eyes"
    cv2.namedWindow(win)
    try:
        cv2.setWindowProperty(win, cv2.WND_PROP_TOPMOST, 1)  # force to front
    except Exception:
        pass
    print("(webcam window open — 6 second preview, then I'll describe it)")
    frame = None
    t0 = time.time()
    while time.time() - t0 < 6:
        ok, f = cam.read()
        if not ok:
            break
        frame = f
        left = int(6 - (time.time() - t0)) + 1
        cv2.putText(frame, str(left), (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow(win, frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # early exit still works
            break
        try:
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        except Exception:
            break
    cv2.destroyAllWindows()
    cam.release()
    if frame is None:
        return "I couldn't open the webcam."
    if float(frame.mean()) < 5:
        return "It's too dark for me to see anything right now."
    # Find faces so the description centers on the person, not invented objects.
    n, best_box, best_face, clean_frame = scan_faces(frame)
    cv2.imwrite(EYE_JPG, frame)  # full frame with boxes, for Michael to check
    # Crop tightly to the person so the vision model can't invent background objects.
    send_img = crop_person(frame, best_box) if best_box is not None else frame
    if best_box is not None:
        print("(cropped to the person for description)")
    print(f"(found {n} face{'s' if n != 1 else ''}, describing...)")
    # Is it Michael? Compare against his enrolled fingerprint.
    recognized = False
    if RECOG_OK and best_face is not None:
        recognized, match_score = is_michael(clean_frame, best_face)
        print(f"(face match: {match_score:.2f} — "
              f"{'thats Michael!' if recognized else 'unknown person'})")
    if recognized:
        task = ("The person in this photo is Michael. Describe him — "
                "appearance, clothing, expression, what he is doing — "
                "in two short sentences, as if greeting someone you know.")
    elif best_box is not None:
        # The model only sees the cropped person — nothing else to invent.
        task = ("Describe the person in this photo — appearance, clothing, "
                "expression, what they are doing — in two short sentences.")
    elif n > 1:
        task = (f"There are {n} people in this photo (green boxes mark faces). "
                "Describe each person briefly — appearance, clothing, what they "
                "are doing — in two or three short sentences. Describe only "
                "what is clearly visible; do not invent other objects.")
    else:
        task = ("No person is clearly visible in this photo. Describe only "
                "what is clearly visible, in one or two short sentences. "
                "Do not guess, imagine, or mention objects that are not "
                "clearly there. If you are unsure, say so.")
    return ask_vision(send_img, task)


def read_mood():
    """Look at the face in front of the camera and read the mood."""
    frame = grab_quick_frame()
    if frame is None:
        return "I couldn't find a working camera."
    if float(frame.mean()) < 5:
        return "It's too dark for me to read your face right now."
    n, best_box, best_face, clean_frame = scan_faces(frame)
    if best_box is None:
        return ("I can't see your face clearly — look at the camera "
                "and ask me again.")
    face_img = crop_head(frame, best_box)
    task = ("Look at this person's face. What emotion do they show? "
            "Start your answer with exactly one of these words: happy, sad, "
            "tired, stressed, surprised, angry, calm, or neutral. "
            "Then add one short warm sentence, like a friend noticing.")
    return ask_vision(face_img, task, num_predict=60)


HARDCODED_ROKU_IP = "10.0.0.78"  # Michael's Roku TV (its network screen, 2026-10-02)


def roku_ping(ip):
    """Read-only check: is there a Roku at this address?"""
    try:
        with urllib.request.urlopen(
                f"http://{ip}:8060/query/device-info", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def roku_find_ip(skip_hardcode=False):
    """Find the Roku on the local Wi-Fi. Tries, in order:
    (1) the roku_ip setting (ping-verified), (2) the hardwired Roku TV
    address (ping-verified), (3) SSDP discovery, (4) a subnet sweep of
    port 8060. Remembers the winner in SETTINGS for this session."""
    manual = SETTINGS.get("roku_ip", "").strip()
    if manual and roku_ping(manual):
        return manual
    if HARDCODED_ROKU_IP and not skip_hardcode and roku_ping(HARDCODED_ROKU_IP):
        print(f"(Roku TV answering at {HARDCODED_ROKU_IP})")
        return HARDCODED_ROKU_IP
    ip = _roku_ssdp() or _roku_sweep()
    if ip:
        SETTINGS["roku_ip"] = ip
        print(f"(Roku found at {ip})")
    return ip


def _roku_ssdp():
    """Ask the network 'who is a Roku?' (multicast). Fast when it works,
    but some routers block it."""
    import socket
    msg = ('M-SEARCH * HTTP/1.1\r\n'
           'HOST: 239.255.255.250:1900\r\n'
           'MAN: "ns=01; ns=01;"\r\n'
           'ST: roku:ecp\r\nMX: 2\r\n\r\n')
    for _ in range(3):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM,
                          socket.IPPROTO_UDP)
        try:
            s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
            s.settimeout(2.0)
            s.sendto(msg.encode(), ("239.255.255.250", 1900))
            while True:
                try:
                    data, addr = s.recvfrom(2048)
                except socket.timeout:
                    break
                if b"roku" in data.lower():
                    return addr[0]
        except Exception as e:
            print(f"(Roku SSDP attempt failed: {e})")
        finally:
            s.close()
    return None


def _roku_sweep():
    """Knock on port 8060 of every address on our subnet and ask
    'are you a Roku?' Slower, but routers can't block it."""
    import socket as _sock
    from concurrent.futures import ThreadPoolExecutor
    try:
        tmp = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        tmp.connect(("8.8.8.8", 80))
        my_ip = tmp.getsockname()[0]
        tmp.close()
    except Exception:
        return None
    if my_ip.startswith("127."):
        return None
    base = ".".join(my_ip.split(".")[:3]) + "."
    print(f"(scanning {base}1-254 for the Roku — a few seconds...)")

    def probe(i):
        ip = base + str(i)
        if ip == my_ip:
            return None
        try:
            with urllib.request.urlopen(
                    f"http://{ip}:8060/query/device-info",
                    timeout=0.5) as r:
                body = r.read(2048).decode(errors="ignore")
                if "roku" in body.lower():
                    return ip
        except Exception:
            return None
        return None

    with ThreadPoolExecutor(max_workers=64) as ex:
        for ip in ex.map(probe, range(1, 255)):
            if ip:
                return ip
    return None


def save_setting(key, value):
    """Persist one KEY=value into nevaeh_settings.txt (newest copy)."""
    path = find_file("nevaeh_settings.txt")
    if not path:
        path = os.path.join(DOWNLOADS, "nevaeh_settings.txt")
    lines = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    out, seen = [], False
    for ln in lines:
        s = ln.strip()
        if (s and not s.startswith("#") and "=" in s
                and s.split("=", 1)[0].strip().lower() == key):
            out.append(f"{key}={value}\n")
            seen = True
        else:
            out.append(ln)
    if not seen:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append(f"{key}={value}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(out)
    SETTINGS[key] = value
    print(f"(saved {key}={value} to {os.path.basename(path)})")


def roku_key(key, ip):
    """Send one remote keypress to the Roku."""
    try:
        req = urllib.request.Request(f"http://{ip}:8060/keypress/{key}",
                                     data=b"", method="POST")
        urllib.request.urlopen(req, timeout=5)
        return True
    except Exception as e:
        print(f"(Roku {key} failed: {e})")
        return False


def roku_apps(ip):
    """List installed Roku apps as (id, name)."""
    try:
        with urllib.request.urlopen(f"http://{ip}:8060/query/apps",
                                    timeout=5) as r:
            xml = r.read().decode()
        return re.findall(r'<app[^>]*id="(\d+)"[^>]*>([^<]+)</app>', xml)
    except Exception:
        return []


def roku_launch(appname, ip):
    """Launch a Roku app by name. Returns the matched app name or None."""
    want = appname.lower()
    for aid, name in roku_apps(ip):
        if want in name.lower() or name.lower() in want:
            try:
                req = urllib.request.Request(
                    f"http://{ip}:8060/launch/{aid}", data=b"", method="POST")
                urllib.request.urlopen(req, timeout=5)
                return name
            except Exception:
                return None
    return None


def roku_is_tv(ip):
    """True if the Roku identifies as a TV (can power on/off).
    A stick/box can only sleep itself, not the TV."""
    try:
        with urllib.request.urlopen(
                f"http://{ip}:8060/query/device-info", timeout=5) as r:
            body = r.read(4096).decode(errors="ignore").lower()
            if "<is-tv>true</is-tv>" in body:
                return True
            if "<is-stick>true</is-stick>" in body:
                return False
            # model names containing "tv" (e.g. Roku TV models)
            m = re.search(r"<model-name>([^<]*)</model-name>", body)
            if m and "tv" in m.group(1):
                return True
    except Exception:
        pass
    return None  # unknown


def roku_send(key):
    """Send a keypress, re-scanning once if the address went stale.
    Returns True if a Roku answered."""
    ip = roku_find_ip()
    if ip and roku_key(key, ip):
        return True
    print("(Roku didn't answer — rescanning the Wi-Fi once...)")
    SETTINGS["roku_ip"] = ""
    ip = roku_find_ip(skip_hardcode=True)
    if ip and roku_key(key, ip):
        SETTINGS["roku_ip"] = ip  # remember the new address
        return True
    return False


def roku_command(heard):
    """Handle TV/Roku voice commands. Returns the spoken reply."""
    ip = roku_find_ip()
    if not ip:
        return ("I scanned the whole Wi-Fi and still can't see your Roku. "
                "Let's do it by hand: on the Roku remote press Home, then go "
                "to Settings, Network, About — read me the IP address "
                "(the four numbers with dots), and I'll remember it forever.")
    t = heard.lower()
    no_reach = ("I couldn't reach your Roku — it may be unplugged or "
                "off the Wi-Fi. Say 'Roku home' to test me when it's on.")
    # --- TV power ---
    if any(p in t for p in ("turn the tv on", "turn on the tv", "tv on",
                            "power on", "turn the television on")):
        if roku_is_tv(ip):
            ok = roku_send("PowerOn") or roku_send("Power")
        else:
            # stick/box: Home wakes a CEC-linked TV
            ok = roku_send("Home")
        return "TV on." if ok else no_reach
    if any(p in t for p in ("turn the tv off", "turn off the tv", "tv off",
                            "power off", "turn the television off")):
        ok = roku_send("PowerOff")
        if not ok:
            return no_reach
        if roku_is_tv(ip) is False:
            return ("That's a Roku stick — it can't turn your TV itself "
                    "off, only the TV's own remote can. I put the Roku "
                    "to sleep.")
        return "TV off."
    # --- volume ---
    if any(p in t for p in ("volume up", "turn the volume up")) or \
            ("turn it up" in t and ("tv" in t or "roku" in t or "volume" in t)):
        ok = any(roku_send("VolumeUp") for _ in range(3))
        return "Volume up." if ok else no_reach
    if any(p in t for p in ("volume down", "turn the volume down")) or \
            ("turn it down" in t and ("tv" in t or "roku" in t or "volume" in t)):
        ok = any(roku_send("VolumeDown") for _ in range(3))
        return "Volume down." if ok else no_reach
    if "mute" in t and ("tv" in t or "roku" in t or "volume" in t):
        ok = roku_send("VolumeMute")
        if not ok:
            return no_reach
        return "Unmuted." if "unmute" in t else "Muted."
    # --- channels ---
    if any(p in t for p in ("channel up", "next channel",
                            "change the channel", "switch the channel")):
        return "Channel up." if roku_send("ChannelUp") else no_reach
    if any(p in t for p in ("channel down", "previous channel")):
        return "Channel down." if roku_send("ChannelDown") else no_reach
    # --- navigation / transport after the word roku ---
    m = re.search(r"\broku\b(.*)", t)
    rest = m.group(1).strip(" ,.!?") if m else ""
    nav = {"home": ("Home", "Home screen."), "up": ("Up", "Up."),
           "down": ("Down", "Down."), "left": ("Left", "Left."),
           "right": ("Right", "Right."), "ok": ("Select", "OK."),
           "select": ("Select", "Selected."), "back": ("Back", "Back."),
           "play": ("Play", "Playing."), "pause": ("Play", "Paused.")}
    if rest in nav:
        key, reply = nav[rest]
        return reply if roku_send(key) else no_reach
    if rest.startswith("launch ") or rest.startswith("open "):
        app = roku_launch(rest.split(" ", 1)[1].strip(), ip)
        return f"Opening {app}." if app else "I couldn't find that app on your Roku."
    return ("I can do: TV on and off, volume up and down, mute, channel up "
            "and down, home, up, down, left, right, OK, back, play, pause, "
            "or 'Roku launch' plus an app name.")


def alexa_plug_command(heard):
    """Route a smart-home phrase through Alexa ('turn on the porch light').

    Amazon plugs only answer through Alexa's cloud, so this needs the
    one-time setup_alexa.py login. Sends the phrase as an Alexa voice
    command, e.g. Alexa, 'turn on the porch light'.
    """
    try:
        import nevaeh_alexa as NALEXA
    except ImportError:
        return "My Alexa helper file is missing — re-download nevaeh_alexa.py."
    if not NALEXA.configured():
        return ("Alexa isn't connected yet — double-click setup_alexa_v3.py, "
                "log into Amazon once, then try again.")
    cmd = heard.strip().rstrip(".!,")
    ok = NALEXA.alexa_say(cmd)
    if ok:
        return f"Okay — {cmd}."
    if ok is None:
        return "My Alexa helper isn't ready — try again in a minute."
    return ("Alexa didn't respond. If this keeps happening, double-click "
            "setup_alexa_v3.py to reconnect it.")




def web_search(query):
    """Free web search via DuckDuckGo instant answers. No API key."""
    try:
        url = ("https://api.duckduckgo.com/?" + urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}))
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        if data.get("AbstractText"):
            return data["AbstractText"][:600]
        for t in data.get("RelatedTopics", []):
            if isinstance(t, dict) and t.get("Text"):
                return t["Text"][:600]
        return ""
    except Exception:
        return ""


def wiki_summary(topic):
    """One-paragraph Wikipedia summary. No API key."""
    try:
        url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
               + urllib.parse.quote(topic))
        req = urllib.request.Request(url,
                                     headers={"User-Agent": "NevaehBrain/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read()).get("extract", "")[:800]
    except Exception:
        return ""


def get_weather():
    """Current + tomorrow weather for Burton, MI. Free, no key (Open-Meteo)."""
    try:
        url = ("https://api.open-meteo.com/v1/forecast?latitude=42.93"
               "&longitude=-83.62&current=temperature_2m,apparent_temperature,"
               "weathercode&daily=temperature_2m_max,temperature_2m_min,"
               "weathercode&temperature_unit=fahrenheit&timezone=America"
               "%2FDetroit&forecast_days=2")
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        wmo = {0: "clear sky", 1: "mostly clear", 2: "partly cloudy",
               3: "overcast", 45: "foggy", 48: "icy fog", 51: "light drizzle",
               53: "drizzle", 55: "heavy drizzle", 61: "light rain",
               63: "rain", 65: "heavy rain", 71: "light snow", 73: "snow",
               75: "heavy snow", 80: "light showers", 81: "showers",
               82: "heavy showers", 95: "thunderstorms"}
        cur, daily = data["current"], data["daily"]
        return (f"Right now in Burton it's {round(cur['temperature_2m'])} "
                f"degrees and {wmo.get(cur['weathercode'], 'unknown')}, "
                f"feels like {round(cur['apparent_temperature'])}. "
                f"Today: high {round(daily['temperature_2m_max'][0])}, "
                f"low {round(daily['temperature_2m_min'][0])}. Tomorrow: high "
                f"{round(daily['temperature_2m_max'][1])}, "
                f"{wmo.get(daily['weathercode'][1], 'unknown')}.")
    except Exception as e:
        return f"Couldn't reach the weather service: {e}"


def remember_fact(text):
    """Append a fact to nevaeh_memory.txt and reload it. 'remember ...'"""
    global MEMORY
    t = re.sub(r"^(please\s+)?remember\s+(that\s+)?", "", text,
               flags=re.IGNORECASE).strip().rstrip(".")
    if not t:
        return "What should I remember?"
    t = t[0].upper() + t[1:]
    path = find_file("nevaeh_memory.txt") or os.path.join(DOWNLOADS,
                                                          "nevaeh_memory.txt")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n- {t}.\n")
        MEMORY = load_memory()
        return f"Remembered: {t}."
    except Exception as e:
        return f"Couldn't save that: {e}"


def set_reminder(text):
    """'remind me in 20 minutes to call mom' — speaks when time is up."""
    m = re.search(r"remind me in (\d+)\s*(minutes?|hours?)", text.lower())
    if not m:
        return "Say it like: remind me in 20 minutes to call mom."
    n, unit = int(m.group(1)), m.group(2)
    secs = n * 3600 if unit.startswith("hour") else n * 60
    parts = re.split(r"remind me in \d+\s*(?:minutes?|hours?)\s*(to\s+)?",
                     text, maxsplit=1, flags=re.IGNORECASE)
    what = parts[1].strip().rstrip(".") if len(parts) > 1 and parts[1].strip() \
        else "your reminder"

    def _wait():
        time.sleep(secs)
        msg = f"Reminder: {what}."
        print(f"\nNEVAEH: {msg}")
        try:
            speak(msg)
        except Exception:
            pass

    threading.Thread(target=_wait, daemon=True).start()
    return f"Got it — I'll remind you in {n} {unit}."


def motion_sensor():
    """Watch the camera for motion; announce strangers, note Michael.
    Say 'stop watching' to end."""
    cam = open_camera(640, 480)
    for _ in range(10):
        cam.read()
    print("(motion sensor ON — say 'stop watching' to turn it off)")
    speak("Motion sensor on.")
    prev, last_listen, last_alert = None, 0.0, 0.0
    try:
        while True:
            ok, frame = cam.read()
            if ok and prev is not None:
                a = cv2.GaussianBlur(
                    cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                b = cv2.GaussianBlur(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
                _, thresh = cv2.threshold(cv2.absdiff(a, b), 25, 255,
                                          cv2.THRESH_BINARY)
                if (thresh > 0).mean() > 0.02:  # 2% of pixels changed
                    handle_motion(frame, last_alert)
                    last_alert = time.time()
            prev = frame
            if time.time() - last_listen > 12:
                last_listen = time.time()
                cmd = listen(timeout_secs=4)
                if cmd and "stop watching" in cmd.lower():
                    break
            time.sleep(0.3)
    finally:
        cam.release()
    speak("Motion sensor off.")
    return "Motion sensor off."


def handle_motion(frame, last_alert):
    """On motion: identify who moved. Strangers get announced."""
    try:
        if not (FACE_OK and RECOG_OK):
            print("(motion detected)")
            return
        h, w = frame.shape[:2]
        FACE_DETECTOR.setInputSize((w, h))
        _, detected = FACE_DETECTOR.detect(frame)
        if detected is None:
            print("(motion detected, no face)")
            return
        for row in detected:
            if float(row[14]) > 0.5:
                match, score = is_michael(frame, row)
                if match:
                    print(f"(motion: Michael, match {score:.2f})")
                elif time.time() - last_alert > 60:
                    print("(motion: unknown person!)")
                    cv2.imwrite(os.path.join(HERE, "motion.jpg"), frame)
                    speak("Someone is here.")
                else:
                    print("(motion: unknown person, alert cooling down)")
                return
        print("(motion detected, face unclear)")
    except Exception as e:
        print(f"(motion check glitch: {e})")


def wants_eyes(text):
    t = text.lower()
    return any(p in t for p in [
        "what do you see", "can you see", "look at", "describe what",
        "what am i holding", "do you see"])


def wants_autostart(text):
    t = text.lower()
    return any(p in t for p in [
        "start with my computer", "run at startup", "start automatically",
        "auto start", "launch at startup"])


def enable_autostart():
    """Copy the launcher into Windows' Startup folder. No typing needed."""
    try:
        startup = os.path.join(os.path.expanduser("~"), "AppData", "Roaming",
                               "Microsoft", "Windows", "Start Menu",
                               "Programs", "Startup")
        src = os.path.join(DOWNLOADS, "start_nevaeh.py")
        if not os.path.exists(src):
            return "I couldn't find the launcher file to copy."
        import shutil
        shutil.copy(src, os.path.join(startup, "start_nevaeh.py"))
        return ("Done — I'll start by myself every time your computer "
                "turns on.")
    except Exception as e:
        return f"I couldn't set that up: {e}"


def michael_present():
    """Quick silent check: is Michael in front of the camera right now?
    No window, no photo saved — just a look."""
    if not (FACE_OK and RECOG_OK) or CAM_IDX is None:
        return False
    try:
        cam = open_camera(640, 480)
        frame = None
        for _ in range(10):
            ok, f = cam.read()
            if ok:
                frame = f
        cam.release()
        if frame is None:
            return False
        h, w = frame.shape[:2]
        FACE_DETECTOR.setInputSize((w, h))
        _, detected = FACE_DETECTOR.detect(frame)
        if detected is None:
            return False
        for row in detected:
            if float(row[14]) > 0.5:
                match, score = is_michael(frame, row)
                if match and score > 0.4:  # stricter bar for auto-greeting
                    return True
        return False
    except Exception:
        return False


def similar(a, b):
    aw, bw = set(a.lower().split()), set(b.lower().split())
    if not aw or not bw:
        return False
    return len(aw & bw) / max(len(aw), len(bw)) > 0.6



# --- secure update channel: vendored Ed25519 verification (no new deps) ---
_BQ = (1 << 255) - 19
_BL = (1 << 252) + 27742317777372353535851937790883648493


def _binv(x):
    return pow(x, _BQ - 2, _BQ)


_BD = (-121665 * _binv(121666)) % _BQ
_BGx = 15112221349535400772501151409588531511454012693041857206046113283949847762202
_BGy = 46316835694926478169428394003475163141307993866256225615783033603165251855960


def _bedwards(p, q):
    x1, y1, x2, y2 = p[0], p[1], q[0], q[1]
    x3 = (x1 * y2 + x2 * y1) * _binv(1 + _BD * x1 * x2 * y1 * y2) % _BQ
    y3 = (y1 * y2 + x1 * x2) * _binv(1 - _BD * x1 * x2 * y1 * y2) % _BQ
    return (x3, y3)


def _bscalarmult(p, e):
    q = (0, 1)
    while e > 0:
        if e & 1:
            q = _bedwards(q, p)
        p = _bedwards(p, p)
        e >>= 1
    return q


def _bxrecover(y):
    xx = (y * y - 1) * _binv(_BD * y * y + 1) % _BQ
    x = pow(xx, (_BQ + 3) // 8, _BQ)
    if (x * x - xx) % _BQ != 0:
        x = (x * pow(2, (_BQ - 1) // 4, _BQ)) % _BQ
    return x


def _bisoncurve(p):
    x, y = p
    return (-x * x + y * y - 1 - _BD * x * x * y * y) % _BQ == 0


def _bdecodepoint(s):
    y = int.from_bytes(s, "little") & ((1 << 255) - 1)
    x = _bxrecover(y)
    if x & 1 != (s[31] >> 7):
        x = _BQ - x
    p = (x, y)
    if not _bisoncurve(p):
        raise ValueError("bad point")
    return p


def _bhint(m):
    return int.from_bytes(hashlib.sha512(m).digest(), "little")


def _bcheckvalid(sig, msg, pub):
    """Raise ValueError unless sig is a valid Ed25519 signature."""
    if len(sig) != 64 or len(pub) != 32:
        raise ValueError("bad lengths")
    if int.from_bytes(sig[32:], "little") >= _BL:
        raise ValueError("bad s")
    a_pt = _bdecodepoint(pub)
    r_pt = _bdecodepoint(sig[:32])
    h = _bhint(sig[:32] + pub + msg)
    if _bscalarmult((_BGx, _BGy), int.from_bytes(sig[32:], "little")) != \
            _bedwards(r_pt, _bscalarmult(a_pt, h)):
        raise ValueError("bad signature")


BRAIN_VERSION = 62
UPDATE_MANIFEST_URL = ("https://raw.githubusercontent.com/"
                       "mcrobertsmichael9-ai/nevaeh-brain/main/version.json")
UPDATE_PUBKEY = bytes.fromhex(
    "b28f891f38038f412d1c9fdb2f242717f32d1b5a1a777e8de4762cbd51e3951b")
_PENDING_UPDATE = None  # (version, brain_bytes, [(name, bytes)]) when staged
last_active = time.time()


def _upd_fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "NevaehBrain"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _upd_verify_manifest(man):
    payload = f"{man['version']}:{man['sha256']}:{man['url']}".encode()
    _bcheckvalid(bytes.fromhex(man["signature"]), payload, UPDATE_PUBKEY)


def stage_brain_update():
    """Download + cryptographically verify a brain update. Returns the new
    version number if one was staged, else None. Never installs unverified."""
    global _PENDING_UPDATE
    try:
        if "NEVAEH_OWNER" in UPDATE_MANIFEST_URL:
            return None
        man = json.loads(_upd_fetch(UPDATE_MANIFEST_URL))
        latest = int(man.get("version", 0))
        if latest <= BRAIN_VERSION:
            return None
        _upd_verify_manifest(man)  # raises unless Jarvis signed it
        brain = _upd_fetch(man["url"])
        if hashlib.sha256(brain).hexdigest() != man["sha256"]:
            print("(update: brain hash mismatch — rejected)")
            return None
        files = []
        for f in man.get("files", []):
            fb = _upd_fetch(f["url"])
            if hashlib.sha256(fb).hexdigest() != f["sha256"]:
                print(f"(update: file {f['name']} hash mismatch — rejected)")
                return None
            files.append((f["name"], fb))
        _PENDING_UPDATE = (latest, brain, files)
        return latest
    except Exception as e:
        print(f"(update check: {e})")
        return None


def install_staged_update():
    """Write the staged update next to this brain and relaunch into it."""
    global _PENDING_UPDATE
    if not _PENDING_UPDATE:
        return False
    ver, brain, files = _PENDING_UPDATE
    _PENDING_UPDATE = None
    try:
        cur = os.path.abspath(__file__)
        d = os.path.dirname(cur)
        for name, fb in files:
            with open(os.path.join(d, name), "wb") as fh:
                fh.write(fb)
            print(f"(updated file: {name})")
        m = re.search(r"\((\d+)\)", os.path.basename(cur))
        n = int(m.group(1)) + 1 if m else ver
        newbase = re.sub(r"\s*\(\d+\)", "", os.path.basename(cur))
        newbase = newbase.replace(".py", f" ({n}).py")
        newpath = os.path.join(d, newbase)
        with open(newpath, "wb") as fh:
            fh.write(brain)
        print(f"(my upgrade is ready: {newbase} — restarting myself)")
        return newpath
    except Exception as e:
        print(f"(update install failed: {e})")
        return False


def update_watcher():
    """Background: check for updates at startup, then every 6 hours."""
    import time as _t
    _t.sleep(45)  # let her finish starting up first
    while True:
        ver = stage_brain_update()
        if ver:
            print(f"(brain v{ver} is staged — I'll switch when we're idle)")
        _t.sleep(6 * 3600)


_FACE_THREAD = None


def self_diagnose():
    """She checks her own health and fixes what she can:
    re-probes mics and switches to a livelier one, verifies her face
    images, checks Ollama/models, and looks for missing files."""
    global MIC, GAIN
    report, fixed = [], []

    # 1) mic — re-probe everything, switch if something better appeared
    try:
        cands = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                try:
                    rec = sd.rec(int(SAMPLE_RATE * 0.5),
                                 samplerate=SAMPLE_RATE, channels=1,
                                 dtype="float32", device=i)
                    sd.wait()
                    lvl = float(np.sqrt(np.mean(np.nan_to_num(
                        rec ** 2, nan=0.0, posinf=1e12))))
                    if lvl != lvl:  # mic sending garbage, not signal
                        lvl = -1.0
                    cands.append((lvl, i, dev["name"]))
                except Exception:
                    continue
        if cands:
            best_level, best, best_name = max(cands)
            report.append(f"Mic: strongest signal {best_level:.6f} "
                          f"({best_name[:40]}).")
            if best_level < 0.0005:
                report.append("Every mic is silent — I can't unmute a mic "
                              "myself, Michael. Tap the mic-mute key on the "
                              "laptop (amber light means muted).")
            elif best != MIC:
                MIC = best
                GAIN = min(max(0.02 / max(best_level, 1e-6), 1.0), 400.0)
                fixed.append(f"switched to a better mic (gain x{GAIN:.1f})")
            else:
                report.append("Mic is fine — I'm already on the best one.")
        else:
            report.append("No microphones found at all.")
    except Exception as e:
        report.append(f"Mic check glitch: {e}.")

    # 2) face images
    n_img = sum(1 for s in ("nevaeh_face", "nevaeh_mouth1", "nevaeh_mouth2")
                for i in range(4) if find_file(f"{s}_{i}.png"))
    has_blink = bool(find_file("nevaeh_blink.png"))
    if n_img == 12 and has_blink:
        report.append("Face: all 13 portrait images found.")
    else:
        report.append(f"Face: only {n_img} of 12 portraits found — "
                      "re-download my face images from the email.")

    # 3) thinking models
    try:
        models = ollama_models() or []
        have_fast = any(n == FAST_MODEL or n.startswith(FAST_MODEL + ":")
                        for n in models)
        have_big = any(n == MODEL or n.startswith(MODEL + ":")
                       for n in models)
        report.append(f"Thinking: Ollama is up. Fast brain "
                      f"{'ready' if have_fast else 'still downloading'}, "
                      f"big brain {'ready' if have_big else 'missing'}.")
        if not have_fast:
            pull_model_bg(FAST_MODEL, label="faster brain")
            fixed.append("started downloading my faster brain")
    except Exception:
        report.append("Thinking: can't reach Ollama — is the Ollama app "
                      "running?")

    # 4) companion files
    for f_ in ("nevaeh_memory.txt", "nevaeh_settings.txt", "irene_data.txt"):
        if not find_file(f_):
            report.append(f"Missing file: {f_} — grab it from my email.")

    # 5) memory
    report.append(f"Memory: {len(MEMORY.splitlines()) if MEMORY else 0} "
                  "notes about you.")

    if fixed:
        report.append("Fixed just now: " + "; ".join(fixed) + ".")
    return " ".join(report)


print("Nevaeh brain v62 online — crash-proof mic. Say 'goodbye' to stop.")
threading.Thread(target=update_watcher, daemon=True).start()
print("(secure update channel on — I check for my own upgrades)")
threading.Thread(target=start_phone_remote, daemon=True).start()
print("Ask 'what do you see?' to use the webcam.")
print("Say 'go to sleep' anytime and I'll wait quietly for 'Nevaeh'.")
print("I'll also greet you myself whenever you walk up to the camera.")
paused = SETTINGS.get("wake_word", "off").lower() == "on"
if paused:
    speak("Hey Michael, I'll hang back quietly till you say my name.")
    print("(paused — say 'Nevaeh' to give a command)")
if SETTINGS.get("type_box", "on").lower() == "on":
    start_type_box()
    print("(type box open — type a command there anytime)")
last_spoken = ""
last_check = 0.0
was_present = False
try:
    while True:
        # Proactive watch: every 30s, silently check the camera. Greet Michael
        # only the moment he appears (not while he's just sitting there).
        # Never when paused — paused means silent.
        if (_PENDING_UPDATE and time.time() - last_active > 60):
            ver = _PENDING_UPDATE[0]
            newpath = install_staged_update()
            if newpath:
                speak(f"I've got my version {ver} upgrade. "
                      "Restarting myself now, Michael.")
                subprocess.Popen([sys.executable, newpath])
                print("(restarting into the new brain)")
                break
        if time.time() - last_check > 30:
            last_check = time.time()
            try:
                if (FACE_STATE.get("visible") and _FACE_THREAD is not None
                        and not _FACE_THREAD.is_alive()):
                    print("(my face window died — restarting it myself)")
                    start_type_box()
                if not paused and SETTINGS.get("greet_me", "on").lower() == "on":
                    present = michael_present()
                    if present and not was_present:
                        speak("Hey Michael! Good to see you.")
                        print("NEVAEH: Hey Michael! Good to see you.")
                        last_spoken = "Hey Michael! Good to see you."
                    was_present = present
            except Exception:
                pass
        heard = None
        is_typed = False
        if paused:
            # Paused: wait for "Nevaeh" (or a typed command), take ONE
            # command, then pause again.
            woke = sleep_listen()
            if isinstance(woke, str) and woke:
                heard, is_typed = woke, True
                print("TYPED:", heard)
            else:
                speak("Hey, I'm here — what do you need?")
                print("NEVAEH: Yes?")
                last_spoken = "Yes?"
                heard = listen()
                if not heard:
                    continue  # back to pause
        else:
            heard = poll_typed()
            if heard is not None:
                is_typed = True
                print("TYPED:", heard)
            else:
                heard = listen()
        if not heard:
            print("(heard nothing, listening again)")
            continue
        if similar(heard, last_spoken):
            print("(ignored my own voice, listening again)")
            continue
        if not is_typed and not voice_is_michael():
            speak("I only take voice orders from Michael.")
            print("NEVAEH: I only take voice orders from Michael.")
            last_spoken = "I only take voice orders from Michael."
            continue
        print("YOU:", heard)
        farewell = heard.lower().strip().rstrip(".!,")
        if farewell in ("goodbye", "goodbye michael", "bye michael"):
            speak("Goodbye, Michael. I'll be right here when you need me.")
            print("NEVAEH: Goodbye, Michael. I'll be right here when you need me.")
            for t in _mem_threads:
                t.join(timeout=15)
            break
        if any(p in heard.lower() for p in ("go to sleep", "sleep now",
                                             "sleep mode on", "take a nap")):
            speak("I'll rest a little, Michael. Just say my name when you want me.")
            print("NEVAEH: I'll rest a little, Michael. Just say my name when you want me.")
            last_spoken = "I'll rest a little, Michael. Just say my name when you want me."
            paused = True
            continue
        streamed = False
        if wants_eyes(heard):
            reply = snap_and_describe()
        elif wants_autostart(heard):
            reply = enable_autostart()
        else:
            t = heard.lower()
            if any(p in t for p in ["enroll my voice", "learn my voice",
                                    "remember my voice"]):
                reply = enroll_my_voice()
            elif any(p in t for p in ["lock my voice", "voice lock on"]):
                if VOICE_ID_AVAILABLE and NVOICE.has_voiceprint():
                    save_setting("voice_lock", "on")
                    reply = ("Voice lock is on — I'll only take orders "
                             "from you now.")
                else:
                    reply = ("I don't know your voice yet — say "
                             "'enroll my voice' first.")
            elif any(p in t for p in ["unlock my voice", "voice lock off"]):
                save_setting("voice_lock", "off")
                reply = "Voice lock is off — I'll answer anyone again."
            elif "voice filter off" in t:
                save_setting("whisper_vad", "off")
                reply = ("Voice filter off — I'll transcribe everything I "
                         "hear, background noise included.")
            elif "voice filter on" in t:
                save_setting("whisper_vad", "on")
                reply = "Voice filter back on."
            elif "sharper ears" in t:
                save_setting("whisper_model", "small")
                reply = ("Sharpest hearing coming up — I'll catch every word, "
                         "but I'll take longer to answer. Restart me.")
            elif "balanced ears" in t:
                save_setting("whisper_model", "base")
                reply = ("Balanced ears — good hearing, faster answers. "
                         "Restart me.")
            elif "faster ears" in t:
                save_setting("whisper_model", "tiny")
                reply = ("Fastest answers coming up — but I might mishear "
                         "more. Restart me.")
            elif any(p in t for p in ["self check", "diagnose yourself",
                                             "check yourself", "diagnose me",
                                             "run diagnostics"]):
                print("(running self-diagnosis...)")
                reply = self_diagnose()
            elif any(p in t for p in ["check for updates", "check for upgrade",
                                             "update yourself", "any updates"]):
                ver = stage_brain_update()
                if ver:
                    reply = (f"Found my version {ver} upgrade — I'll switch "
                             "over when we're done talking.")
                elif _PENDING_UPDATE:
                    reply = "My upgrade is already downloaded and waiting."
                else:
                    reply = "I'm all up to date, Michael."
            elif "roll your eyes" in t or "eye roll" in t:
                FACE_STATE["expression"] = ("eyeroll", time.time() + 4)
                reply = "Oh, absolutely."
            elif "give me a wink" in t or t.strip() == "wink" or "wink at me" in t:
                FACE_STATE["expression"] = ("wink", time.time() + 3)
                reply = "There. Don't tell anyone."
            elif t.strip() == "smile" or "give me a smile" in t:
                FACE_STATE["expression"] = ("smile", time.time() + 4)
                reply = "Always, for you."
            elif "flirt" in t:
                FACE_STATE["expression"] = ("flirt", time.time() + 5)
                reply = "Is it working?"
            elif "tic tac toe" in t or "tick tack toe" in t:
                play_tictactoe()
                reply = "Played tic-tac-toe with Michael."
                streamed = True
            elif any(p in t for p in ["start with the computer",
                                             "start on startup",
                                             "auto start", "start on boot",
                                             "launch on startup"]):
                reply = setup_autostart()
            elif any(p in t for p in ["don't start with the computer",
                                     "stop auto start",
                                     "don't auto start"]):
                reply = remove_autostart()
            elif "show phone link" in t or "phone link" in t:
                FACE_STATE["qr_until"] = time.time() + 60
                ip = _nevaeh_lan_ip()
                reply = (f"Scan the code on my face, or open "
                         f"http colon slash slash {ip} colon 8765 "
                         f"on your phone.")
            elif "face off" in t:
                face_set_visible(False)
                reply = "My face is hidden — say 'face on' to bring me back."
            elif "face on" in t:
                face_set_visible(True)
                reply = "I'm here — can you see me?"
            elif ("full screen" in t or "fullscreen" in t
                  or "big face" in t or "maximize" in t):
                FACE_STATE["fullscreen"] = True
                FACE_STATE["_rebuild"] = True
                save_setting("face_size", "full")
                reply = "Going big — say 'small face' anytime to shrink me back."
            elif "small face" in t or "window face" in t:
                FACE_STATE["fullscreen"] = False
                FACE_STATE["_rebuild"] = True
                save_setting("face_size", "window")
                reply = "Back in my little window."
            elif "irene" in t and any(p in t for p in
                                      ["evaluate all", "evaluate everyone",
                                       "run evaluation", "run the evaluation"]):
                reply = irene_evaluate_all()
            elif t.strip().rstrip(".!,") in ("irene", "run irene",
                                             "irene evaluate"):
                reply = irene_evaluate_all()
            elif "irene" in t and "evaluat" in t:
                name = t.split("evaluat", 1)[1]
                for w in ("irene", "the creator", "creator"):
                    name = name.replace(w, "")
                reply = irene_evaluate_one(name.strip(" .!,"))
            elif any(p in t for p in ["motion sensor on", "start motion sensor",
                                   "watch for motion"]):
                reply = motion_sensor()
            elif "weather" in t:
                reply = get_weather()
            elif any(p in t for p in ["read my mood", "how do i look",
                                      "how am i feeling", "what's my mood",
                                      "what is my mood", "how's my mood"]):
                reply = read_mood()
            elif "roku" in t and "ip" in t:
                # Teach it the Roku's address by voice, before anything else
                # Roku-related runs.
                m = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", heard)
                if m:
                    save_setting("roku_ip", m.group(1))
                    reply = (f"Got it — your Roku is at {m.group(1)}. "
                             f"I'll remember it from now on.")
                else:
                    reply = ("Tell me the four numbers with dots, like "
                             "'my Roku is at 192.168.1.20'.")
            elif "roku" in t or any(p in t for p in
                                    ["turn the tv on", "turn on the tv",
                                     "turn the tv off", "turn off the tv",
                                     "turn the television on",
                                     "turn the television off",
                                     "volume up", "volume down",
                                     "turn the volume up",
                                     "turn the volume down",
                                     "channel up", "channel down",
                                     "change the channel",
                                     "switch the channel", "mute the tv"]):
                reply = roku_command(heard)
            elif any(p in t for p in ["turn on the ", "turn off the ",
                                      "switch on the ", "switch off the ",
                                      "dim the "]):
                # Amazon smart plugs/lights via Alexa (after TV checks above)
                reply = alexa_plug_command(heard)
            elif t.startswith("remember"):
                reply = remember_fact(heard)
            elif "remind me in" in t:
                reply = set_reminder(heard)
            elif "use phone camera" in t:
                if SETTINGS.get("phone_camera", "").strip():
                    SETTINGS["camera"] = "phone"
                    reply = "Phone camera it is."
                else:
                    reply = ("I don't have your phone's camera address yet. "
                             "Put it in nevaeh_settings.txt as phone_camera="
                             "http://your-phone-ip:8080/video")
            elif "use laptop camera" in t:
                SETTINGS["camera"] = "laptop"
                reply = "Back to the laptop camera."
            elif any(p in t for p in ["open live studio", "open tiktok studio",
                                      "launch live studio", "start live studio",
                                      "open tiktok live studio",
                                      "launch tiktok live studio"]):
                reply = open_live_studio()
            elif any(p in t for p in ["plan my live", "plan tonight's live",
                                      "plan tonights live", "live stream ideas",
                                      "stream ideas", "live stream plan"]):
                reply = think_and_speak(
                    "Help me plan tonight's TikTok live stream: give me a "
                    "hook for the first 30 seconds, 3 segment ideas, and one "
                    "engagement tactic. Two short sentences per idea, max.",
                    tiktok=True)
                streamed = True
            elif any(p in t for p in ["search for", "search the web",
                                      "look up", "google "]):
                q = re.sub(r".*?(search for|search the web|look up|google)\s+",
                           "", heard, flags=re.IGNORECASE).strip()
                hit = web_search(q) or wiki_summary(q)
                txt = (f"{heard}\n\nWeb result: {hit}\n\n"
                       "Answer in 1-2 sentences.") if hit else heard
                reply = think_and_speak(txt)
                streamed = True
            elif t.startswith("who is ") or t.startswith("what is "):
                hit = wiki_summary(heard)
                txt = (f"{heard}\n\nReference: {hit}\n\n"
                       "Answer in 1-2 sentences.") if hit else heard
                reply = think_and_speak(txt)
                streamed = True
            elif any(p in t for p in ["tiktok", "live stream", "streaming",
                                      "go live", "my live", "streamer",
                                      "battle", "pk"]):
                reply = think_and_speak(heard, tiktok=True)
                streamed = True
            else:
                reply = think_and_speak(heard)
                streamed = True
        if not streamed:
            print("NEVAEH:", reply)
            speak(reply)
        last_spoken = reply
        last_active = time.time()
        if heard and heard.strip():
            remember_turn(heard, reply)
except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    import traceback
    print(f"\nSomething went wrong: {e}")
    traceback.print_exc()
    input("Press Enter to close this window.")
