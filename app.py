import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import joblib
from gtts import gTTS
import pygame
from translate import Translator
import threading
import os
import time
import re

class FilmceptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Filmception - Movie AI Processor")
        self.root.geometry("800x600")
        
        pygame.mixer.init()
        self.models_loaded = False
        self.load_models_async()
        self.create_loading_screen()
    
    def load_models_async(self):
        def load_task():
            try:
                start_time = time.time()
                self.model = joblib.load("models/genre_model.pkl")  # Full pipeline
                self.mlb = joblib.load("models/mlb.pkl")
                load_time = time.time() - start_time
                print(f"Models loaded in {load_time:.2f} seconds")
                self.root.after(0, self.create_main_ui)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror(
                    "Error", 
                    f"Failed to load models:\n{str(e)}"
                ))
                self.root.after(0, self.root.destroy)
        threading.Thread(target=load_task, daemon=True).start()
    
    def create_loading_screen(self):
        self.loading_frame = ttk.Frame(self.root)
        self.loading_frame.pack(expand=True, fill=tk.BOTH)
        
        ttk.Label(
            self.loading_frame, 
            text="Loading Filmception...", 
            font=('Helvetica', 16)
        ).pack(pady=50)
        
        self.progress = ttk.Progressbar(
            self.loading_frame, 
            mode='indeterminate'
        )
        self.progress.pack(pady=10)
        self.progress.start()
    
    def create_main_ui(self):
        self.loading_frame.destroy()
        self.models_loaded = True
        
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Enter Movie Summary:").pack(anchor='w')
        self.summary_text = scrolledtext.ScrolledText(
            main_frame, height=10, wrap=tk.WORD, font=('Helvetica', 10)
        )
        self.summary_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.predict_btn = ttk.Button(
            btn_frame, text="Predict Genres", command=self.start_prediction, state='normal'
        )
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        trans_frame = ttk.Frame(btn_frame)
        trans_frame.pack(side=tk.LEFT, padx=20)
        
        ttk.Label(trans_frame, text="Language:").pack(anchor='w')
        self.lang_var = tk.StringVar(value='en')
        lang_menu = ttk.Combobox(
            trans_frame, textvariable=self.lang_var,
            values=['en', 'ar', 'ur', 'ko'], state='readonly', width=7
        )
        lang_menu.pack()
        
        self.audio_btn = ttk.Button(
            trans_frame, text="Play Audio", command=self.start_audio_translation, state='normal'
        )
        self.audio_btn.pack(pady=5)
        
        ttk.Label(main_frame, text="Results:").pack(anchor='w')
        self.results_text = scrolledtext.ScrolledText(
            main_frame, height=8, wrap=tk.WORD, font=('Helvetica', 10), state='disabled'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)
        
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            main_frame, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor='w'
        )
        status_bar.pack(fill=tk.X, pady=(10, 0))
    
    def set_busy_state(self, busy=True):
        state = 'disabled' if busy else 'normal'
        self.predict_btn.config(state=state)
        self.audio_btn.config(state=state)
        self.status_var.set("Processing..." if busy else "Ready")
        self.root.update()
    
    def clean_text(self, text):
        return re.sub(r"[^a-zA-Z\s]", "", text.lower())
    
    def start_prediction(self):
        if not self.models_loaded:
            messagebox.showerror("Error", "Models not loaded yet")
            return
        
        summary = self.summary_text.get("1.0", tk.END).strip()
        if not summary:
            messagebox.showwarning("Warning", "Please enter a movie summary")
            return
        
        self.set_busy_state(True)
        
        def prediction_task():
            try:
                cleaned = self.clean_text(summary)
                y_pred = self.model.predict([cleaned])
                genres = self.mlb.inverse_transform(y_pred)[0]

                result = "Predicted Genres:\n"
                result += ", ".join(genres) if genres else "No genres predicted"

                self.results_text.config(state='normal')
                self.results_text.delete("1.0", tk.END)
                self.results_text.insert(tk.END, result)
                self.results_text.config(state='disabled')
            except Exception as e:
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            finally:
                self.set_busy_state(False)

        threading.Thread(target=prediction_task, daemon=True).start()
    
    def start_audio_translation(self):
        if not self.models_loaded:
            messagebox.showerror("Error", "Models not loaded yet")
            return
        
        summary = self.summary_text.get("1.0", tk.END).strip()
        if not summary:
            messagebox.showwarning("Warning", "Please enter a movie summary")
            return
        
        lang = self.lang_var.get()
        self.set_busy_state(True)
        
        def audio_task():
            try:
                # Generate unique filename for each translation
                audio_file = f"temp_audio_{lang}_{int(time.time())}.mp3"
                
                # Clean up any previous audio files
                for f in os.listdir():
                    if f.startswith("temp_audio_") and f.endswith(".mp3"):
                        try:
                            os.remove(f)
                        except:
                            pass
                
                # Split long text into chunks
                max_chars = 450
                text_chunks = [summary[i:i+max_chars] for i in range(0, len(summary), max_chars)]
                
                translator = Translator(to_lang=lang)
                translated_chunks = []
                
                for chunk in text_chunks:
                    translated = translator.translate(chunk)
                    translated_chunks.append(translated)
                    time.sleep(1)  # Avoid rate limiting
                
                full_translation = " ".join(translated_chunks)
                
                # Update results
                result = f"Translation ({lang}):\n{full_translation}"
                self.results_text.config(state='normal')
                self.results_text.delete("1.0", tk.END)
                self.results_text.insert(tk.END, result)
                self.results_text.config(state='disabled')

                # Generate and play audio
                tts = gTTS(text=full_translation, lang=lang)
                tts.save(audio_file)
                
                if pygame.mixer.music.get_busy():
                    pygame.mixer.music.stop()
                
                pygame.mixer.music.load(audio_file)
                time.sleep(0.2)  # Ensure file is fully loaded
                pygame.mixer.music.play()
                
            except Exception as e:
                messagebox.showerror("Error", f"Translation failed: {str(e)}")
            finally:
                self.set_busy_state(False)

        threading.Thread(target=audio_task, daemon=True).start()

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
        messagebox.showwarning(
            "Warning", 
            "Created 'models' directory\n"
            "Please ensure model files are present:\n"
            "- genre_model.pkl\n"
            "- mlb.pkl"
        )
    
    root = tk.Tk()
    app = FilmceptionApp(root)
    root.eval('tk::PlaceWindow . center')
    root.mainloop()
