import streamlit as st
import pandas as pd
import tempfile
import pickle
import os
import time
import core

st.set_page_config(page_title="Zapptain America", layout="wide", initial_sidebar_state="collapsed")

# ----------------- PRO-LEVEL CYBERPUNK CSS -----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp { background-color: #0E1117; color: #C9D1D9; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #E6EDF3; font-weight: 800; letter-spacing: -0.5px; }
    
    /* Headers & Text */
    .app-title { font-size: 42px; font-weight: 800; color: #00f2fe; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0; }
    .subtitle { color: #8B949E; font-size: 13px; font-weight: 700; letter-spacing: 1.5px; margin-bottom: 25px; text-transform: uppercase; }
    .step-header { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #00f2fe; letter-spacing: 2px; text-transform: uppercase; margin-top: 40px;}
    .desc-text { color: #8B949E; font-size: 15px; line-height: 1.6; margin-bottom: 20px;}
    
    /* Telemetry Metrics Box */
    div[data-testid="metric-container"] {
        background-color: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: 'JetBrains Mono', monospace;
    }
    div[data-testid="metric-container"] > label { color: #8B949E !important; font-size: 11px !important; letter-spacing: 1px;}
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #00f2fe !important; font-size: 24px !important; font-weight: bold;}
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { color: #E6EDF3 !important; font-size: 12px !important;}
    
    /* Match Hero Box */
    .match-hero {
        background: linear-gradient(145deg, #161b22 0%, #0d1117 100%);
        border-left: 5px solid #ffa500; border-radius: 6px; padding: 30px; margin-top: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    .match-hero-title { color: #ffa500; font-size: 12px; font-weight: 800; letter-spacing: 2px; margin-bottom: 5px; }
    .match-song { font-size: 36px; font-weight: 800; color: #ffffff; line-height: 1.2; margin-bottom: 10px;}
    .match-stats { font-family: 'JetBrains Mono', monospace; color: #8B949E; font-size: 14px; }
    .highlight-stat { color: #00f2fe; font-weight: bold; }
    
    /* Candidate Leaderboard */
    .leaderboard-row { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 14px; margin-bottom: 4px;}
    .bar-bg { background-color: #21262D; width: 100%; height: 6px; border-radius: 3px; margin-bottom: 15px; overflow: hidden;}
    .bar-fill { background-color: #388bfd; height: 100%; border-radius: 3px; transition: width 0.5s ease-in-out;}
    
    /* Library Cards */
    .lib-card {
        background-color: #161b22; border: 1px solid #30363D; border-radius: 8px; padding: 15px;
        margin-bottom: 15px; text-align: center; transition: all 0.2s ease;
    }
    .lib-card:hover { border-color: #00f2fe; transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0, 242, 254, 0.15); }
    .lib-song { font-weight: 600; color: #E6EDF3; font-size: 16px; margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
    .lib-hash { font-family: 'JetBrains Mono', monospace; color: #8B949E; font-size: 12px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='app-title'>⚡ ZAPPTAIN AMERICA</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>EE200 Audio Fingerprinting • Advanced Identification Matrix</div>", unsafe_allow_html=True)

@st.cache_resource
def load_db():
    if os.path.exists("database.pkl"):
        with open("database.pkl", "rb") as f:
            data = pickle.load(f)
            # Handle both new dictionary format and old raw db format
            if isinstance(data, dict) and 'hashes' in data:
                return data['hashes'], data.get('metadata', {})
            return data, {}
    return {}, {}

db, db_metadata = load_db()

tab1, tab2, tab3 = st.tabs(["[ LIBRARY ]", "[ IDENTIFY ]", "[ BATCH RUN ]"])

# ----------------- TAB 1: LIBRARY -----------------
with tab1:
    st.markdown("### INDEXED TARGETS")
    if not db:
        st.error("Index missing. Please run `build_index.py` locally and upload `database.pkl`.")
    else:
        # Sort songs alphabetically for the grid
        songs = sorted(list(db_metadata.keys())) if db_metadata else sorted(list(set(s for hashes in db.values() for s, _ in hashes)))
        st.markdown("<div class='desc-text'>Database loaded. The system has extracted structural fingerprints from the following audio files.</div>", unsafe_allow_html=True)
        
        cols = st.columns(4)
        for i, song in enumerate(songs):
            hash_count = db_metadata.get(song, "Unknown")
            with cols[i % 4]:
                st.markdown(f"""
                <div class='lib-card'>
                    <div class='lib-song'>{song}</div>
                    <div class='lib-hash'>[ {hash_count:,} HASHES ]</div>
                </div>
                """, unsafe_allow_html=True)

# ----------------- TAB 2: IDENTIFY -----------------
with tab2:
    col_left, col_right = st.columns([1, 3])
    
    with col_left:
        st.markdown("### QUERY INPUT")
        uploaded_file = st.file_uploader("Upload Audio (WAV/MP3/M4A)", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'], label_visibility="collapsed")
        
        # Audio augmentation options (Satisfies Q3A robustness testing)
        st.markdown("### SIGNAL AUGMENTATION")
        add_noise = st.checkbox("Inject White Noise", help="Test robustness against background interference")
        pitch_shift = st.checkbox("Pitch Shift (+1 step)", help="Test structural integrity under pitch alterations")
        
        if uploaded_file:
            st.audio(uploaded_file)
            run_btn = st.button("INITIALIZE TRACE", type="primary", use_container_width=True)
    
    with col_right:
        if uploaded_file is not None and run_btn:
            if not db:
                st.error("Database is empty. Index required.")
            else:
                with st.spinner("Analyzing spectral envelope..."):
                    # 1. Save temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # 2. Process with Telemetry Logging
                    t0 = time.time()
                    y, sr = core.load_audio(tmp_path)
                    
                    # Apply augmentations if selected
                    if add_noise:
                        y = core.apply_noise(y)
                    if pitch_shift:
                        y = core.apply_pitch_shift(y, sr)
                    
                    t1 = time.time()
                    f, t, Sxx = core.get_spectrogram(y, sr)
                    
                    t2 = time.time()
                    peak_f, peak_t = core.extract_peaks(Sxx)
                    
                    t3 = time.time()
                    hashes = core.generate_hashes(peak_f, peak_t)
                    
                    t4 = time.time()
                    ranked_songs, all_matches = core.match_query(hashes, db)
                    t5 = time.time()
                    
                    # 3. Render Telemetry Dashboard
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("SPECTROGRAM", f"{int((t2-t1)*1000)} ms", "FFT Engine")
                    m2.metric("CONSTELLATION", f"{int((t3-t2)*1000)} ms", f"{len(peak_f):,} peaks")
                    m3.metric("HASHING", f"{int((t4-t3)*1000)} ms", f"{len(hashes):,} anchors")
                    m4.metric("DB LOOKUP", f"{int((t5-t4)*1000)} ms", "O(1) Access")
                    m5.metric("TOTAL LATENCY", f"{int((t5-t0)*1000)} ms", "Complete")
                    
                    # 4. Result Evaluation
                    if ranked_songs and ranked_songs[0][1] > 10:
                        best_song, best_score = ranked_songs[0]
                        runner_up_score = ranked_songs[1][1] if len(ranked_songs) > 1 else 1
                        ratio = best_score / max(1, runner_up_score)
                        
                        # Hero Banner
                        st.markdown(f"""
                        <div class='match-hero'>
                            <div class='match-hero-title'>MATCH CONFIRMED</div>
                            <div class='match-song'>{best_song}</div>
                            <div class='match-stats'>
                                ALIGNMENT SCORE: <span class='highlight-stat'>{best_score:,}</span> | 
                                CONFIDENCE RATIO: <span class='highlight-stat'>{ratio:.1f}x</span> HIGHER THAN RUNNER-UP
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Leaderboard
                        lead_col1, lead_col2 = st.columns([1, 1.5])
                        with lead_col1:
                            st.markdown("<div class='subtitle'>CANDIDATE LEADERBOARD</div>", unsafe_allow_html=True)
                            for song, score in ranked_songs[:5]:
                                pct = min(100, (score / best_score) * 100)
                                st.markdown(f"""
                                <div class='leaderboard-row'><span>{song}</span> <span>{score:,}</span></div>
                                <div class='bar-bg'><div class='bar-fill' style='width: {pct}%;'></div></div>
                                """, unsafe_allow_html=True)
                        
                        # Explanations & Plots
                        st.markdown("<div class='step-header'>STEP 1: FEATURE EXTRACTION</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='desc-text'>The audio signal is mapped into a time-frequency domain. By discarding amplitude and phase dynamics, the system achieves robustness against EQ shifts and noise. The engine isolated <b>{len(peak_f):,}</b> dominant local maxima (the constellation).</div>", unsafe_allow_html=True)
                        st.pyplot(core.plot_constellation(Sxx, f, t, peak_f, peak_t))
                        
                        st.markdown("<div class='step-header'>STEP 2: THE ALIGNMENT SPIKE</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='desc-text'>Every matched structural hash casts a vote for a specific time-offset between the query and the database track. False positives scatter randomly, creating a low noise floor. A genuine match forces the offsets to converge synchronously. <b>{best_score:,}</b> structural pairs aligned perfectly at a single moment in time.</div>", unsafe_allow_html=True)
                        st.plotly_chart(core.plot_alignment_spike(all_matches, best_song), use_container_width=True)

                    else:
                        st.error("No definitive match found. The clip may be corrupted beyond threshold, or it does not exist in the Zapptain index.")

# ----------------- TAB 3: BATCH RUN -----------------
with tab3:
    st.markdown("### BATCH PROCESSING PROTOCOL")
    st.markdown("<div class='desc-text'>Upload a directory of audio snippets. The system will iterate through the dataset and output a verified <code>results.csv</code> file.</div>", unsafe_allow_html=True)
    
    batch_files = st.file_uploader("Upload Multiple Queries", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'], accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("EXECUTE BATCH", type="primary") and batch_files:
        if not db:
             st.error("Database is empty. Index required.")
        else:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(batch_files):
                status_text.text(f"Processing: {file.name}")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                    
                y, sr = core.load_audio(tmp_path)
                f, t, Sxx = core.get_spectrogram(y, sr)
                peak_f, peak_t = core.extract_peaks(Sxx)
                hashes = core.generate_hashes(peak_f, peak_t)
                
                ranked_songs, _ = core.match_query(hashes, db)
                
                # Per PDF instructions: prediction is the matched song's filename without extension
                if ranked_songs and ranked_songs[0][1] > 10:
                    prediction = os.path.splitext(ranked_songs[0][0])[0] 
                else:
                    prediction = "NONE"
                
                results.append({"filename": file.name, "prediction": prediction})
                progress_bar.progress((i + 1) / len(batch_files))
                
            status_text.empty()
            
            if results:
                st.success(f"Batch protocol complete. {len(results)} files processed.")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("DOWNLOAD RESULTS.CSV", data=csv, file_name="results.csv", mime="text/csv", type="primary")