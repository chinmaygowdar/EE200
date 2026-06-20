import os
import pickle
import core

AUDIO_DIR = "song_database"
DB_FILE = "database.pkl"

def build_database():
    if not os.path.exists(AUDIO_DIR):
        print(f"Error: Directory '{AUDIO_DIR}' not found. Please create it and add audio files.")
        return

    db = {}
    metadata = {}
    
    print("Initializing Zapptain America Indexing Sequence...")
    print("Extracting high-density fingerprints. This will take a moment...\n")
    
    total_hashes = 0
    
    for file in os.listdir(AUDIO_DIR):
        if file.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
            song_name = os.path.splitext(file)[0]
            filepath = os.path.join(AUDIO_DIR, file)
            
            y, sr = core.load_audio(filepath)
            f, t, Sxx = core.get_spectrogram(y, sr)
            peak_f, peak_t = core.extract_peaks(Sxx)
            hashes = core.generate_hashes(peak_f, peak_t)
            
            hash_count = len(hashes)
            metadata[song_name] = hash_count
            total_hashes += hash_count
            
            print(f"Processed: {song_name.ljust(30)} | Peaks: {len(peak_f):<6} | Hashes: {hash_count:<6}")
            
            for h, t1 in hashes:
                if h not in db:
                    db[h] = []
                db[h].append((song_name, t1))
                
    # Save a wrapped dictionary containing both the DB and the Metadata for the UI
    export_data = {
        'hashes': db,
        'metadata': metadata
    }
                
    with open(DB_FILE, 'wb') as f:
        pickle.dump(export_data, f)
        
    db_size_mb = os.path.getsize(DB_FILE) / (1024 * 1024)
    print(f"\n[SUCCESS] Index built and saved to {DB_FILE}")
    print(f"Total Unique Hashes: {len(db)}")
    print(f"Total Hashes Indexed: {total_hashes}")
    print(f"Database Size: {db_size_mb:.2f} MB")

if __name__ == "__main__":
    build_database()