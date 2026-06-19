import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import Counter
import matplotlib.pyplot as plt
import librosa

def load_audio(file_path, sr=22050):
    """Loads audio file and returns mono signal."""
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def get_spectrogram(y, sr):
    """Computes the spectrogram of the audio."""
    f, t, Sxx = signal.spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
    Sxx_log = 10 * np.log10(Sxx + 1e-10) # Convert to dB
    return f, t, Sxx_log

def extract_peaks(Sxx_log, threshold=-40, neighborhood_size=20):
    """Finds local maxima in the spectrogram (Constellation Map)."""
    # Find local maxima
    local_max = maximum_filter(Sxx_log, size=neighborhood_size) == Sxx_log
    background = (Sxx_log > threshold)
    eroded_background = np.logical_and(local_max, background)
    
    # Get peak coordinates
    peak_freq_idx, peak_time_idx = np.where(eroded_background)
    
    # Keep only the strongest peaks to optimize speed
    peak_amps = Sxx_log[peak_freq_idx, peak_time_idx]
    sort_idx = np.argsort(peak_amps)[::-1][:800] 
    
    return peak_freq_idx[sort_idx], peak_time_idx[sort_idx]

def generate_hashes(peak_f, peak_t, num_neighbors=5):
    """Generates structural hashes from pairs of peaks."""
    hashes = []
    points = sorted(zip(peak_t, peak_f)) # Sort by time
    
    for i in range(len(points)):
        t1, f1 = points[i]
        for j in range(1, num_neighbors + 1):
            if i + j < len(points):
                t2, f2 = points[i+j]
                dt = t2 - t1
                if 0 < dt < 100: # Restrict time gap
                    h = f"{f1}|{f2}|{dt}"
                    hashes.append((h, t1))
    return hashes

def match_query(query_hashes, db):
    """Finds the best matching song based on the alignment spike."""
    matches = []
    for h, t1 in query_hashes:
        if h in db:
            for song_name, db_t1 in db[h]:
                offset = db_t1 - t1
                matches.append((song_name, offset))
                
    if not matches:
        return None, 0, []
        
    # Find the song and offset that appear most frequently
    match_counts = Counter(matches)
    best_match, max_score = match_counts.most_common(1)[0]
    best_song = best_match[0]
    
    # Extract all offsets for the winning song to plot the histogram
    best_offsets = [offset for (song, offset) in matches if song == best_song]
    
    return best_song, max_score, best_offsets

def plot_constellation(Sxx, f, t, peak_f, peak_t):
    """Visualizer for Step 1."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma', alpha=0.5)
    ax.scatter(t[peak_t], f[peak_f], c='cyan', s=15, marker='x')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig

def plot_offset_histogram(matches_list):
    """Visualizer for Step 2."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(matches_list, bins=100, color='#00f2fe', edgecolor='black')
    ax.set_xlabel('Time Offset (Database frames - Query frames)')
    ax.set_ylabel('Number of Aligning Hashes')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    return fig