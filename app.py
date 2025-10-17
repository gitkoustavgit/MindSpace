import time
import os
import re
import requests
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
# voice_input is still available if you later add a CLI mode
from voice_input import get_voice_input  # optional; not used in web mode
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from google.api_core.exceptions import ResourceExhausted, NotFound, PermissionDenied, FailedPrecondition

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------- Load Environment Variables ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")  # optional; agentic tool works only if provided

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# ---------- Configure Gemini ----------
genai.configure(api_key=api_key)

# Prefer stable 1.5 models; explicitly avoid experimental and 2.5 for now
PREFERRED_SUBSTRINGS = [
    "1.5-flash-002",
    "1.5-flash",
    "1.5-flash-8b",
    "1.0-pro",           # wide availability on older accounts
    "pro"                # last resort (but NOT experimental)
]

def _is_experimental(name: str) -> bool:
    n = name.lower()
    return ("exp" in n) or ("experimental" in n) or ("2.5" in n)

def _supports_generate(m) -> bool:
    methods = getattr(m, "supported_generation_methods", None) or []
    return "generateContent" in methods

def _rank_model_name(name: str) -> int:
    name_l = name.lower()
    for i, sub in enumerate(PREFERRED_SUBSTRINGS):
        if sub in name_l:
            return i
    return len(PREFERRED_SUBSTRINGS) + 1

def _pick_model_name():
    """Return a stable, non-experimental model name available to the key/project."""
    try:
        models = list(genai.list_models())
    except Exception:
        # If listing fails, try known stable IDs directly (API accepts either short id or 'models/<id>')
        return "models/gemini-1.5-flash-002"

    usable = []
    for m in models:
        name = getattr(m, "name", "")
        if not name:
            continue
        if _is_experimental(name):
            continue
        if not _supports_generate(m):
            continue
        usable.append(name)

    if not usable:
        # Hard fallbacks
        return "models/gemini-1.5-flash-002"

    usable.sort(key=_rank_model_name)
    return usable[0]

# Initialize global model (will be swapped if we hit 429/404)
_CURRENT_MODEL_NAME = _pick_model_name()
gemini_model = genai.GenerativeModel(_CURRENT_MODEL_NAME)

def _swap_to_next_model():
    """Pick the next preferred model different from current and rebuild gemini_model."""
    global _CURRENT_MODEL_NAME, gemini_model
    # Make a preference-ordered list of candidates
    candidates = [
        "models/gemini-1.5-flash-002",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-flash-8b",
        "models/gemini-1.0-pro",
        "models/gemini-pro",
    ]
    # Ensure uniqueness and drop experimental/2.5
    seen = set()
    cleaned = []
    for c in candidates:
        if not _is_experimental(c) and c not in seen:
            seen.add(c); cleaned.append(c)
    # Merge any other listable, usable models
    try:
        lm = [m.name for m in genai.list_models() if _supports_generate(m) and not _is_experimental(m.name)]
        for name in lm:
            if name not in seen:
                seen.add(name); cleaned.append(name)
    except Exception:
        pass

    # rotate to the next after current
    if _CURRENT_MODEL_NAME in cleaned:
        idx = cleaned.index(_CURRENT_MODEL_NAME)
        next_idx = (idx + 1) % len(cleaned)
    else:
        next_idx = 0

    new_name = cleaned[next_idx]
    _CURRENT_MODEL_NAME = new_name
    gemini_model = genai.GenerativeModel(new_name)
    return new_name

def safe_generate(prompt: str, retries: int = 3):
    """
    Generate with automatic handling of:
    - Quota/Rate (429 ResourceExhausted): swap to another stable model and retry.
    - NotFound/Permission issues: swap models.
    """
    last_err = None
    for attempt in range(retries):
        try:
            resp = gemini_model.generate_content(prompt)
            return resp.text
        except (ResourceExhausted, NotFound, PermissionDenied, FailedPrecondition) as e:
            last_err = e
            # Swap model and retry
            _swap_to_next_model()
            # brief jitter to respect any retry-after header behaviour
            time.sleep(0.6 + 0.2 * attempt)
        except Exception as e:
            # Non-retryable / unexpected
            last_err = e
            break
    # If all attempts failed, surface a clear message (won't crash your app)
    return ("I'm having trouble generating a response right now. "
            "Let's try again in a moment, or rephrase your concern briefly.")

# ---------- Emotion Detection (LAZY-LOADED) ----------
# Avoid importing transformers at startup (slow on Windows).
_sentiment_pipe = None
def get_sentiment_analyzer():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        from transformers import pipeline as hf_pipeline
        _sentiment_pipe = hf_pipeline(
            "sentiment-analysis",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
    return _sentiment_pipe

# ---------- Conversation Memory (short-term session) ----------
conversation_history = []
emotion_history = []
MAX_MEMORY = 5  # keep last 5 turns

# ---------- Prompt Router ----------
def build_prompt(entry: str, emotions: str, repeated_emotion: str = None) -> str:
    history_text = ""
    if conversation_history:
        history_text = "Conversation so far:\n" + "\n".join(conversation_history[-MAX_MEMORY:]) + "\n\n"

    entry_lower = entry.lower()

    # If repeated strong emotions → more detailed guidance
    if repeated_emotion:
        intensity_note = f"\nNote: The user has been expressing repeated {repeated_emotion}. Provide a more detailed, compassionate coping strategy (3–5 sentences)."
    else:
        intensity_note = ""

    if any(word in entry_lower for word in ["exam", "test", "study", "college", "school", "assignment"]):
        return (
            "You are MindSpace, a supportive AI wellness guide for students.\n\n"
            f"{history_text}"
            f"User concern: {entry}\n"
            f"Detected emotions: {emotions}\n"
            f"{intensity_note}\n\n"
            "Task: Suggest ONE well-defined, practical coping strategy for study or exam stress.\n"
            "- Make it concrete (e.g., chunk revision, relaxation breaks, a schedule).\n"
            "- Normal response: 2–4 sentences. If repeated emotions → 3–5 sentences.\n"
            "- Output only the strategy.\n"
        )
    elif any(word in entry_lower for word in ["work", "job", "office", "career", "deadline"]):
        return (
            "You are MindSpace, a supportive AI wellness guide for professionals.\n\n"
            f"{history_text}"
            f"User concern: {entry}\n"
            f"Detected emotions: {emotions}\n"
            f"{intensity_note}\n\n"
            "Task: Suggest ONE coping strategy for workplace stress.\n"
            "- Make it concrete (e.g., prioritization, relaxation during breaks, short walk).\n"
            "- Normal response: 2–4 sentences. If repeated emotions → 3–5 sentences.\n"
            "- Output only the strategy.\n"
        )
    elif any(word in entry_lower for word in ["friend", "family", "relationship", "partner", "parents"]):
        return (
            "You are MindSpace, a supportive AI wellness guide for personal relationships.\n\n"
            f"{history_text}"
            f"User concern: {entry}\n"
            f"Detected emotions: {emotions}\n"
            f"{intensity_note}\n\n"
            "Task: Suggest ONE coping strategy for relationship/family stress.\n"
            "- Examples: active listening, journaling before conversation, setting boundaries.\n"
            "- Normal response: 2–4 sentences. If repeated emotions → 3–5 sentences.\n"
            "- Output only the strategy.\n"
        )
    else:
        return (
            "You are MindSpace, a supportive AI wellness guide.\n\n"
            f"{history_text}"
            f"User concern: {entry}\n"
            f"Detected emotions: {emotions}\n"
            f"{intensity_note}\n\n"
            "Task: Suggest ONE specific, evidence-based coping strategy.\n"
            "- Must be practical (mindfulness, grounding, journaling, social support).\n"
            "- Normal response: 2–4 sentences. If repeated emotions → 3–5 sentences.\n"
            "- Output only the strategy.\n"
        )

# ---------- Function to generate coping strategy ----------
def suggest_coping_strategy(entry: str, emotions: str, repeated_emotion: str = None) -> str:
    prompt = build_prompt(entry, emotions, repeated_emotion)
    generated = safe_generate(prompt) or ""
    generated = generated.strip()

    sentences = generated.split(". ")
    # Expand allowed length if repeated emotion
    max_sentences = 5 if repeated_emotion else 4
    cleaned = ". ".join(sentences[:max_sentences]).strip()
    if cleaned and not cleaned.endswith("."):
        cleaned += "."
    if not cleaned:
        cleaned = "Let’s take this step by step. Start with one small, doable action toward relief."

    # Save conversation context
    conversation_history.append(f"User: {entry}\nAssistant: {cleaned}")
    return cleaned

# ---------- Agentic Video Tool ----------
VIDEO_QUERY_PATTERNS = [
    (r"\b(breath|breathe|breathing|panic|anxiety)\b", "5 minute guided breathing for anxiety"),
    (r"\b(grounding|5-4-3-2-1)\b", "5-4-3-2-1 grounding exercise guided"),
    (r"\b(mindful|mindfulness|meditat)\b", "10 minute guided mindfulness meditation for stress"),
    (r"\b(progressive muscle|pmr|tense and release)\b", "progressive muscle relaxation guided"),
    (r"\b(sleep|insomnia|can't sleep|cant sleep)\b", "sleep relaxation body scan 10 minutes"),
    (r"\b(journal|journaling)\b", "guided journaling for anxiety prompt"),
    (r"\b(cbt|cognitive|thought record|reframe)\b", "CBT thought record tutorial step by step"),
    (r"\b(worry|overthink|rumination)\b", "cognitive defusion exercise guided"),
    (r"\b(study|exam|test|focus|concentrat)\b", "pomodoro timer study with me 25 minute"),
    (r"\b(work|burnout|break|office|deadline)\b", "2 minute desk breathing box breathing"),
]

def _first_match(text: str, patterns):
    t = (text or "").lower()
    for pat, query in patterns:
        if re.search(pat, t):
            return query
    return None

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _yt_search(query: str, max_results: int = 5):
    """YouTube Data API v3: search + get durations. Returns list of dicts. Safe no-op if no API key."""
    if not YOUTUBE_API_KEY:
        return []
    try:
        # 1) search
        search_r = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max_results,
                "safeSearch": "strict",
                "videoEmbeddable": "true",
                "key": YOUTUBE_API_KEY
            },
            timeout=8,
        )
        search_r.raise_for_status()
        items = search_r.json().get("items", [])
        if not items:
            return []

        video_ids = ",".join([it["id"]["videoId"] for it in items])

        # 2) fetch durations/metrics
        vid_r = requests.get(
            "https://www.googleapis.com/youtube/v3/videos",
            params={
                "part": "contentDetails,statistics",
                "id": video_ids,
                "key": YOUTUBE_API_KEY
            },
            timeout=8,
        )
        vid_r.raise_for_status()
        meta_by_id = {it["id"]: it for it in vid_r.json().get("items", [])}

        def parse_iso8601_duration(d):
            # PT#H#M#S, PT#M#S, PT#S
            hours = minutes = seconds = 0
            m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d or "")
            if m:
                hours = int(m.group(1) or 0)
                minutes = int(m.group(2) or 0)
                seconds = int(m.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds

        results = []
        for it in items:
            vid = it["id"]["videoId"]
            snip = it["snippet"]
            meta = meta_by_id.get(vid, {})
            dur_iso = (meta.get("contentDetails") or {}).get("duration", "PT0S")
            dur_sec = parse_iso8601_duration(dur_iso)
            views = int((meta.get("statistics") or {}).get("viewCount", 0))
            results.append({
                "video_id": vid,
                "title": _clean_text(snip.get("title", "")),
                "channel": snip.get("channelTitle", ""),
                "duration_sec": dur_sec,
                "views": views,
                "url": f"https://www.youtube.com/watch?v={vid}",
            })
        return results
    except Exception:
        return []

def _rank_videos(videos, target_min=180, target_max=1200):
    """Prefer short-to-medium, higher views."""
    def score(v):
        dur = v.get("duration_sec", 0) or 0
        in_band = target_min <= dur <= target_max
        mid = (target_min + target_max) / 2
        length_penalty = abs(dur - mid)
        return (1000000 if in_band else 0) + v.get("views", 0) - length_penalty
    return sorted(videos, key=score, reverse=True)

def maybe_recommend_video(user_text: str, strategy_text: str):
    """
    Decide whether to fetch a video.
    - Trigger if user_text or strategy mentions a known activity.
    - Return dict with video info OR None.
    """
    query = _first_match(user_text, VIDEO_QUERY_PATTERNS) or _first_match(strategy_text, VIDEO_QUERY_PATTERNS)
    if not query:
        return None

    candidates = _yt_search(query, max_results=6)
    if not candidates:
        return None

    ranked = _rank_videos(candidates)
    top = ranked[0]

    # final gate: avoid too short (<60s) or too long (>40m)
    if top["duration_sec"] < 60 or top["duration_sec"] > 2400:
        return None

    # consumption guidance tailored to the query
    guide = "Play the video in a quiet spot. Follow along in real time and pause if needed."
    q = query.lower()
    if "breath" in q or "box" in q:
        guide = "Sit upright, shoulders relaxed. Inhale 4, hold 4, exhale 4 (or as guided). Follow the instructor’s pacing."
    elif "grounding" in q:
        guide = "Follow 5-4-3-2-1: name 5 things you see, 4 touch, 3 hear, 2 smell, 1 taste. Let the guide pace you."
    elif "mindful" in q or "meditation" in q or "body scan" in q:
        guide = "Keep eyes soft or closed. If your mind wanders, gently return to the breath/body without judgment."
    elif "progressive muscle" in q:
        guide = "Follow ‘tense then release’ instructions. Don’t strain; aim for gentle tension and full release."
    elif "cbt" in q or "thought record" in q:
        guide = "Keep a notebook ready. Pause to write each step: situation, thoughts, evidence for/against, reframe."
    elif "study" in q or "pomodoro" in q:
        guide = "Use the video as a 25-minute focus block. Before starting, list 1–2 concrete tasks to finish."

    return {
        "title": top["title"],
        "url": top["url"],
        "channel": top["channel"],
        "duration_sec": top["duration_sec"],
        "how_to_consume": guide
    }

# ---------- Function: Speak text out loud ----------
def speak_text(text: str):
    tts = gTTS(text)
    filename = "response.mp3"
    tts.save(filename)
    audio = AudioSegment.from_file(filename, format="mp3")
    play(audio)

# ---------- ChromaDB Setup ----------
chroma_client = chromadb.Client()
try:
    journal_collection = chroma_client.get_or_create_collection(name="mindspace_journals")
except Exception:
    # fallback for older clients
    try:
        journal_collection = chroma_client.create_collection(name="mindspace_journals")
    except Exception:
        journal_collection = chroma_client.get_collection(name="mindspace_journals")

# ---------- Main Analysis ----------
def analyze_journal_entry(entry_text: str, user_id: str = "default_user"):
    # LAZY load emotion analyzer here
    sentiment_analyzer = get_sentiment_analyzer()

    emotions_result = sentiment_analyzer(entry_text)
    emotions = [e["label"] for e in emotions_result]
    emotions_str = ", ".join(emotions)

    # Track emotion history
    emotion_history.append(emotions[0])
    repeated_emotion = None
    if len(emotion_history) >= 2 and emotion_history[-1] == emotion_history[-2]:
        repeated_emotion = emotion_history[-1]

    strategy = suggest_coping_strategy(entry_text, emotions_str, repeated_emotion)

    # Agentic step: optionally attach a video recommendation
    video = maybe_recommend_video(entry_text, strategy)

    metadata = {
        "user": user_id,
        "strategy": strategy,
        "emotions": emotions_str
    }
    if video:
        metadata.update({
            "video_title": video["title"],
            "video_url": video["url"],
            "video_channel": video["channel"],
            "video_duration_sec": video["duration_sec"]
        })

    journal_collection.add(
        documents=[entry_text],
        metadatas=[metadata],
        ids=[f"{user_id}_{int(time.time()*1000)}"]
    )
    return emotions, strategy, video

# ---------- Flask Web Server ----------
# Serve files from ./frontend (where your index.html lives)
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

@app.get("/")
def serve_index():
    # Serves frontend/index.html
    return send_from_directory("frontend", "index.html")

@app.post("/analyze")
def analyze_endpoint():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    allow_videos = bool(data.get("allowVideos", True))

    if not text:
        return jsonify({"emotions": [], "strategy": "Could you share a bit more?", "video": None}), 200

    emotions, strategy, video = analyze_journal_entry(text)

    if not allow_videos:
        video = None

    resp = {
        "emotions": emotions,
        "strategy": strategy,
        "video": video  # either dict or None
    }
    return jsonify(resp), 200

if __name__ == "__main__":
    print("Starting MindSpace web server at http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
