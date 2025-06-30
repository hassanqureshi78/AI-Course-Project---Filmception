import pandas as pd
from translate import Translator  # Changed from googletrans
from gtts import gTTS
import os
import time  # Added import for time module
from tqdm import tqdm

# Load cleaned summaries
df = pd.read_csv("Dataset/test_cleaned.csv")

# Initialize translator
translator = Translator(to_lang="en")  # Default target language

# Define languages and their codes
languages = {
    'arabic': 'ar',
    'urdu': 'ur',
    'korean': 'ko'
}

# Prepare columns for translations
for lang in languages:
    df[f'translated_summary_{lang}'] = ''

# Create audio output directory
audio_dir = "tts_audio"
os.makedirs(audio_dir, exist_ok=True)

print("🌐 Translating and generating TTS for Arabic, Urdu, and Korean.")

# Process first 50 rows
for idx, row in tqdm(df.iloc[:50].iterrows(), total=50):
    summary = row['cleaned_summary']
    movie_id = row['movie_id']

    for lang, lang_code in languages.items():
        try:
            # Translate summary
            translator = Translator(to_lang=lang_code)
            translated = translator.translate(summary)
            df.at[idx, f'translated_summary_{lang}'] = translated

            # Generate TTS audio
            tts = gTTS(text=translated, lang=lang_code)
            audio_path = os.path.join(audio_dir, f"{movie_id}_{lang}.mp3")
            tts.save(audio_path)
            time.sleep(1)  # Add delay to avoid rate limiting

        except Exception as e:
            print(f"⚠️ Error on row {idx} (movie_id={movie_id}) for {lang}: {e}")
            continue

# Save the translations
df.to_csv("Dataset/test_translated.csv", index=False)
print("✅ Translation complete! Saved to test_translated.csv")