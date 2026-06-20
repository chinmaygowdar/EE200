import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter
from collections import Counter
import matplotlib.pyplot as plt
import librosa
import plotly.graph_objects as go

def load_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr, mono=True)
    return y, sr

def apply_noise(y, noise_factor=0.05):
    """Adds white noise to the audio signal for robustness testing."""
    noise = np.random.randn(len(y))
    return y + noise_factor * noise

def apply_pitch_shift(y, sr, n_steps=1):
    """Shifts the pitch of the audio signal."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def get_spectrogram(y, sr):
    f, t, Sxx = signal.spectrogram(y, fs=sr, nperseg=1024, noverlap=512)
    Sxx_log = 10 * np.log10(Sxx + 1e-10) 
    return f, t, Sxx_log

def extract_peaks(Sxx_log, neighborhood_size=15, threshold_drop=35):
    """Finds local maxima. Limits relaxed to capture ~10,000 peaks for a denser DB."""
    threshold = np.max(Sxx_log) - threshold_drop 
    local_max = maximum_filter(Sxx_log, size=neighborhood_size) == Sxx_log
    background = (Sxx_log > threshold)
    eroded_background = np.logical_and(local_max, background)
    
    peak_freq_idx, peak_time_idx = np.where(eroded_background)
    peak_amps = Sxx_log[peak_freq_idx, peak_time_idx]
    
    # Dramatically increased cap to generate the 90-100MB database
    num_peaks = min(len(peak_amps), 10000) 
    if num_peaks == 0:
        return np.array([]), np.array([])
        
    sort_idx = np.argsort(peak_amps)[::-1][:num_peaks] 
    return peak_freq_idx[sort_idx], peak_time_idx[sort_idx]

def generate_hashes(peak_f, peak_t, num_neighbors=15):
    """Generates structural hashes using an expanded target zone."""
    hashes = []
    if len(peak_f) == 0:
        return hashes
        
    points = sorted(zip(peak_t, peak_f)) 
    
    for i in range(len(points)):
        t1, f1 = points[i]
        # Increased neighbors for extreme density and robustness
        for j in range(1, num_neighbors + 1):
            if i + j < len(points):
                t2, f2 = points[i+j]
                dt = t2 - t1
                if 0 < dt < 200: # Wider time-gap tolerance
                    h = f"{f1}|{f2}|{dt}"
                    hashes.append((h, t1))
    return hashes

def match_query(query_hashes, db):
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
        
    song_offsets = {}
    for song, offset in matches:
        if song not in song_offsets:
            song_offsets[song] = []
        song_offsets[song].append(offset)
        
    song_scores = {}
    for song, offsets in song_offsets.items():
        counts = Counter(offsets)
        best_offset, count = counts.most_common(1)[0]
        song_scores[song] = count
        
    ranked_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_songs, matches

# --- ADVANCED PLOTTING FUNCTIONS ---

def plot_constellation(Sxx, f, t, peak_f, peak_t):
    """High-contrast matplotlib spectrogram with neon constellation."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Darker magma for pure cyberpunk feel
    c = ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma', alpha=0.8)
    
    if len(peak_t) > 0:
        ax.scatter(t[peak_t], f[peak_f], c='#00f2fe', s=12, alpha=0.9, edgecolors='black', linewidths=0.5)
        
    ax.set_ylabel('Frequency (Hz)', color='#8B949E')
    ax.set_xlabel('Time (s)', color='#8B949E')
    ax.tick_params(colors='#8B949E')
    
    # Match the Streamlit background
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#0E1117')
    for spine in ax.spines.values():
        spine.set_color('#30363D')
        
    plt.tight_layout()
    return fig

def plot_alignment_spike(matches_list, best_song):
    """Interactive Plotly histogram with custom neon spike annotation."""
    best_offsets = [offset for (song, offset) in matches_list if song == best_song]
    
    if not best_offsets:
        return go.Figure()
        
    # Calculate histogram manually to style the maximum bin
    counts, bins = np.histogram(best_offsets, bins=150)
    max_bin_idx = np.argmax(counts)
    
    colors = ['#00f2fe'] * len(counts)  # Default neon cyan
    colors[max_bin_idx] = '#ffa500'      # Highlight winning spike in neon orange
    
    fig = go.Figure(data=[go.Bar(
        x=bins[:-1], 
        y=counts, 
        marker_color=colors,
        marker_line_width=0,
        width=(bins[1]-bins[0]) * 1.05
    )])
    
    fig.add_annotation(
        x=bins[max_bin_idx],
        y=counts[max_bin_idx],
        text=f"<b>{int(counts[max_bin_idx])} hashes align here</b>",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor="#ffa500",
        font=dict(color="#ffa500", size=14),
        ax=60,
        ay=-40
    )
    
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font=dict(color='#8B949E'),
        xaxis=dict(title='Time Offset (db_frame - query_frame)', showgrid=False, zeroline=False),
        yaxis=dict(title='Hash Count', showgrid=True, gridcolor='#21262D'),
        margin=dict(l=40, r=40, t=40, b=40),
        height=350
    )
    return fig