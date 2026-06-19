import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import Counter
import matplotlib.pyplot as plt
import librosa

def load_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def get_spectrogram(y, sr):
    f, t, Sxx = signal.spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
    Sxx_log = 10 * np.log10(Sxx + 1e-10) 
    return f, t, Sxx_log

def extract_peaks(Sxx_log, neighborhood_size=15, threshold_drop=40):
    """Finds local maxima. Keeps all peaks within a dynamic threshold."""
    threshold = np.max(Sxx_log) - threshold_drop 
    local_max = maximum_filter(Sxx_log, size=neighborhood_size) == Sxx_log
    background = (Sxx_log > threshold)
    eroded_background = np.logical_and(local_max, background)
    
    peak_freq_idx, peak_time_idx = np.where(eroded_background)
    return peak_freq_idx, peak_time_idx

def generate_hashes(peak_f, peak_t, num_neighbors=15):
    """Generates structural hashes from pairs of peaks."""
    hashes = []
    if len(peak_f) == 0:
        return hashes
        
    points = sorted(zip(peak_t, peak_f)) 
    
    for i in range(len(points)):
        t1, f1 = points[i]
        for j in range(1, num_neighbors + 1):
            if i + j < len(points):
                t2, f2 = points[i+j]
                dt = t2 - t1
                if 0 < dt < 100: # Time delta constraint
                    h = f"{f1}|{f2}|{dt}"
                    hashes.append((h, t1))
    return hashes

def match_query(query_hashes, db):
    """Finds the best matching songs and returns ranked candidate scores."""
    if not query_hashes:
        return [], []
        
    matches = []
    for h, t1 in query_hashes:
        if h in db:
            for song_name, db_t1 in db[h]:
                offset = db_t1 - t1
                matches.append((song_name, offset))
                
    if not matches:
        return [], []
        
    # Group offsets by song
    song_offsets = {}
    for song, offset in matches:
        if song not in song_offsets:
            song_offsets[song] = []
        song_offsets[song].append(offset)
        
    # Score is the highest number of aligned hashes at a single offset
    song_scores = {}
    for song, offsets in song_offsets.items():
        counts = Counter(offsets)
        best_offset, count = counts.most_common(1)[0]
        song_scores[song] = count
        
    ranked_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_songs, matches

# --- PLOTTING FUNCTIONS ---
plt.style.use('dark_background')

def plot_constellation(Sxx, f, t, peak_f, peak_t):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma', alpha=0.6)
    if len(peak_t) > 0:
        ax.scatter(t[peak_t], f[peak_f], c='#00f2fe', s=10, alpha=0.8)
    ax.set_ylabel('freq bin')
    ax.set_xlabel('time (s)')
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    plt.tight_layout()
    return fig

def plot_alignment_spike(matches_list, best_song):
    """Plots the alignment histogram for the winning song."""
    fig, ax = plt.subplots(figsize=(12, 4))
    best_offsets = [offset for (song, offset) in matches_list if song == best_song]
    
    if best_offsets:
        counts, bins, patches = ax.hist(best_offsets, bins=150, color='#2b3c4e')
        max_bin = np.argmax(counts)
        patches[max_bin].set_facecolor('#ffa500') # Highlight the spike in orange
        
        ax.annotate(f'{int(counts[max_bin])} hashes\nalign here', 
                    xy=(bins[max_bin], counts[max_bin]), 
                    xytext=(bins[max_bin] + (max(bins)*0.05), counts[max_bin] * 0.8),
                    color='#ffa500', weight='bold',
                    arrowprops=dict(facecolor='#ffa500', edgecolor='#ffa500', arrowstyle='->'))

    ax.set_xlabel('time offset (database frame - query frame)')
    ax.set_ylabel('# hashes')
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    plt.tight_layout()
    return fig