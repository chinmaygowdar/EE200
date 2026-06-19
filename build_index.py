import os
import pickle
import core

AUDIO_DIR = "song_database"
DB_FILE = "database.pkl"

def build_database():
    db = {}
    print("Indexing songs. This might take a few minutes depending on your library size...")
    
    for file in os.listdir(AUDIO_DIR):
        if file.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            song_name = os.path.splitext(file)[0]
            filepath = os.path.join(AUDIO_DIR, file)
            
            y, sr = core.load_audio(filepath)
            f, t, Sxx = core.get_spectrogram(y, sr)
            peak_f, peak_t = core.extract_peaks(Sxx)
            hashes = core.generate_hashes(peak_f, peak_t)
            
            print(f"Processed: {song_name} | {len(peak_f)} peaks | {len(hashes)} hashes")
            
            for h, t1 in hashes:
                if h not in db:
                    db[h] = []
                db[h].append((song_name, t1))
                
    with open(DB_FILE, 'wb') as f:
        pickle.dump(db, f)
    print(f"\nSuccess! Saved {len(db)} unique hashes to {DB_FILE}.")

if __name__ == "__main__":
    build_database()