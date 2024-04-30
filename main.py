import tkinter as tk
from tkinter import filedialog
import assemblyai as aai
import threading
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def to_text(file_path, progress_label, summarize_button):
    aai.settings.api_key = "0b37b233d8a943f6a4ddcdfa55fe3f27"

    progress_label.config(text="Transcribing...")
    config = aai.TranscriptionConfig(
        speaker_labels=True,
    )
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path,config)

    f = open('test.txt','w')

    if transcript.status == aai.TranscriptStatus.error:
        progress_label.config(text="Transcribing Error")
        f.write(transcript.error)
    else:
        progress_label.config(text="Transcribing Complete")
        f.write(transcript.text)
        for utterance in transcript.utterances:
            f.write(f"Speaker {utterance.speaker}: {utterance.text}\n")
        # Enable the summarize button
        summarize_button.config(state=tk.NORMAL)

    f.close()

def file_selected_label(label):
    label.config(text="File Selected")

def generate_pointwise_summary(text):
    # Tokenize the text into sentences
    sentences = text.split(". ")
    
    # Generate bullet points for each sentence
    summary = ""
    for sentence in sentences:
        if sentence.strip() != "":
            summary += "• " + sentence.strip() + ".\n"
    
    return summary

def model_summary():
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained("C:\\ML\\Meeting Notes\\Pegasus-Model-20240408T200935Z-002\\Pegasus-Model", use_safetensors=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("C:\\ML\\Meeting Notes\\Tokenizer\\Tokenizer", tokenizer_config="C:\\ML\\Meeting Notes\\Tokenizer\\Tokenizer\\tokenizer_config.json")

    # Create summarization pipeline
    summarization_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)

    # Open the text file in read mode
    with open('test.txt', 'r') as file:
        # Read the contents of the file into a string
        full_text = file.read()

    # Split the full text into chunks of 1024 tokens
    chunk_size = 1024
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]

    # Generate summary for each chunk
    generated_summaries = []
    for chunk in chunks:
        summary = summarization_pipeline(chunk)[0]["summary_text"]
        generated_summaries.append(summary)

    # Concatenate the generated summaries
    final_summary = "\n".join(generated_summaries)
    final_summary = generate_pointwise_summary(final_summary)
    # Print the final summary
    return final_summary

def to_summary():
    # Implement your summary logic here
    summary = model_summary()
    print("Summarized transcribed text:")
    save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
    
    if save_path:
        # Write the summary to the selected file
        with open(save_path, 'w') as f:
            f.write(summary)
        print("Summary saved successfully.")
    else:
        print("No file selected for saving.")

    print(summary)

class AudioFileDropGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Meeting Notes")

        # Create a frame to hold the drop area
        self.drop_frame = tk.Frame(root, width=400, height=200, bd=1, relief="solid")
        self.drop_frame.pack_propagate(False)
        self.drop_frame.pack(padx=10, pady=10)

        # Create a label for the drop area
        self.drop_label = tk.Label(self.drop_frame, text="Click to select an audio file", bg="lightgray", fg="black", font=("Arial", 12), pady=80)
        self.drop_label.pack(fill=tk.BOTH, expand=True)

        self.selected_file_path = None

        # Allow drop events
        self.drop_label.bind("<Enter>", self.on_enter)
        self.drop_label.bind("<Leave>", self.on_leave)

        # Bind click event to select file
        self.drop_label.bind("<Button-1>", lambda event: self.select_audio_file(event, self.drop_label))

        # Create Go button
        self.go_button = tk.Button(root, text="Go", command=self.process_selected_file)
        self.go_button.pack(pady=10)

        # Create label to show progress
        self.progress_label = tk.Label(root, text="", font=("Arial", 12))
        self.progress_label.pack(pady=5)

        # Create Summarize button (disabled by default)
        self.summarize_button = tk.Button(root, text="Summarize", command=self.summarize_transcript, state=tk.DISABLED)
        self.summarize_button.pack(pady=5)

    def on_enter(self, event):
        self.drop_label.config(bg="lightblue")

    def on_leave(self, event):
        self.drop_label.config(bg="lightgray")

    def select_audio_file(self, event, label):
        self.selected_file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3;*.m4a;*.ogg")])
        if self.selected_file_path:
            file_selected_label(label)

    def process_selected_file(self):
        if self.selected_file_path:
            # Create a new thread to perform transcription
            threading.Thread(target=to_text, args=(self.selected_file_path, self.progress_label, self.summarize_button)).start()
        else:
            print("No audio file selected.")

    def summarize_transcript(self):
        # Implement the summary logic here
        print("Summarizing transcribed text...")
        # Call to_summary function with transcribed text
        to_summary()

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioFileDropGUI(root)
    root.mainloop()

