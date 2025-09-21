import time
import chromadb
from transformers import pipeline
import google.generativeai as genai
import os
from dotenv import load_dotenv
from voice_input import get_voice_input   
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play

# ---------- Load Environment Variables ----------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# ---------- Configure Gemini ----------
genai.configure(api_key=api_key)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------- Emotion Detection ----------
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="j-hartmann/emotion-english-distilroberta-base"
)

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
    response = gemini_model.generate_content(prompt)
    generated = response.text.strip()

    sentences = generated.split(". ")
    # Expand allowed length if repeated emotion
    max_sentences = 5 if repeated_emotion else 4
    cleaned = ". ".join(sentences[:max_sentences]).strip()
    if not cleaned.endswith("."):
        cleaned += "."

    # Save conversation context
    conversation_history.append(f"User: {entry}\nAssistant: {cleaned}")
    return cleaned

# ---------- Function: Speak text out loud ----------
def speak_text(text: str):
    tts = gTTS(text)
    filename = "response.mp3"
    tts.save(filename)
    audio = AudioSegment.from_file(filename, format="mp3")
    play(audio)

# ---------- ChromaDB Setup ----------
chroma_client = chromadb.Client()
journal_collection = chroma_client.create_collection(name="mindspace_journals")

# ---------- Main Analysis ----------
def analyze_journal_entry(entry_text: str, user_id: str = "default_user"):
    emotions_result = sentiment_analyzer(entry_text)
    emotions = [e["label"] for e in emotions_result]
    emotions_str = ", ".join(emotions)

    # Track emotion history
    emotion_history.append(emotions[0])
    repeated_emotion = None
    if len(emotion_history) >= 2 and emotion_history[-1] == emotion_history[-2]:
        repeated_emotion = emotion_history[-1]

    strategy = suggest_coping_strategy(entry_text, emotions_str, repeated_emotion)

    journal_collection.add(
        documents=[entry_text],
        metadatas=[{
            "user": user_id,
            "strategy": strategy,
            "emotions": emotions_str
        }],
        ids=[f"{user_id}_{int(time.time()*1000)}"]
    )
    return emotions, strategy

# ---------- Interactive Run ----------
if __name__ == "__main__":
    print("Welcome to MindSpace 🌱 (type 'exit' anytime to quit)\n")
    print("Choose input method:\n1. Keyboard\n2. Voice")
    choice = input("Enter choice (1/2): ")

    voice_mode = (choice == "2")

    while True:
        if voice_mode:
            # print("\n🎙️ Recording... Speak now")
            entry = get_voice_input(duration=6)
            # print("Recording finished")
            if not entry:
                print("Voice input failed, switching to keyboard.")
                entry = input("What's on your mind? ")
                voice_mode = False
        else:
            entry = input("\nWhat's on your mind? ")

        if not entry:
            continue  # Skip empty input safely

        if entry.lower() in ["exit", "quit", "bye"]:
            print("Goodbye 👋 Stay well!")
            break

        emotions, strategy = analyze_journal_entry(entry)
        print("\n--- MindSpace Response ---")
        print("Detected Emotions:", emotions)
        print("Suggested Coping Strategy:", strategy)

        # If voice input was used, auto-speak the output
        if voice_mode:
            speak_text(strategy)
