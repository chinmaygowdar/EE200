import streamlit as st
import pandas as pd
import tempfile
import pickle
import os
import time
import core

st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS to match the PDF look and feel
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #C9D1D9; }
    h1 { color: #58A6FF; font-weight: 600; margin-bottom: 0px; }
    .subtitle { color: #8B949E; font-size: 14px; font-weight: bold; letter-spacing: 1px; margin-bottom: 20px;}
    .desc { color: #8B949E; font-size: 14px; margin-bottom: 30px;}
    .match-found { color: #238636; font-size: 24px; font-weight: bold; }
    .score-bar-bg { background-color: #21262D; width: 100%; height: 8px; border-radius: 4px; margin-top: 8px;}
    .score-bar-fill { background-color: #58A6FF; height: 8px; border-radius: 4px;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🎧 EE200: Audio Fingerprinting</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>SIGNALS, SYSTEMS & NETWORKS • PROJECT DEMO</div>", unsafe_allow_html=True)
st.markdown("<div class='desc'>Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</div>", unsafe_allow_html=True)

@st.cache_resource
def load_db():
    if os.path.exists("database.pkl"):
        with open("database.pkl", "rb") as f:
            return pickle.load(f)
    return {}

db = load_db()

tab1, tab2, tab3 = st.tabs(["LIBRARY", "IDENTIFY", "BATCH"])

# ----------------- TAB 1: LIBRARY -----------------
with tab1:
    st.subheader("LIBRARY")
    if not db:
        st.error("Index missing. Please run `build_index.py` locally and upload `database.pkl`.")
    else:
        songs = list(set(s for hashes in db.values() for s, _ in hashes))
        st.write("Song indexing is managed by the admin. Drop a clip in the Identify tab to test the library.")
        cols = st.columns(4)
        for i, song in enumerate(songs):
            with cols[i % 4]:
                st.info(f"**{song}**")

# ----------------- TAB 2: IDENTIFY -----------------
with tab2:
    st.subheader("Identify a clip")
    uploaded_file = st.file_uploader("Upload 200MB per file • WAV, MP3, FLAC, OGG", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Identify", type="primary"):
            if not db:
                st.error("Database is empty.")
            else:
                with st.spinner("Processing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    t0 = time.time()
                    y, sr = core.load_audio(tmp_path)
                    
                    t1 = time.time()
                    f, t, Sxx = core.get_spectrogram(y, sr)
                    
                    t2 = time.time()
                    peak_f, peak_t = core.extract_peaks(Sxx)
                    
                    t3 = time.time()
                    hashes = core.generate_hashes(peak_f, peak_t)
                    
                    t4 = time.time()
                    ranked_songs, all_matches = core.match_query(hashes, db)
                    t5 = time.time()
                    
                    # Simulated UI stat row
                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.metric("SPECTROGRAM", f"{int((t2-t1)*1000)} ms")
                    c2.metric("CONSTELLATION", f"{int((t3-t2)*1000)} ms", f"{len(peak_f)} peaks")
                    c3.metric("HASHING", f"{int((t4-t3)*1000)} ms", f"{len(hashes)} hashes")
                    c4.metric("DB LOOKUP", f"{int((t5-t4)*1000)} ms")
                    c5.metric("TOTAL TIME", f"{int((t5-t0)*1000)} ms")
                    
                    st.divider()
                    
                    if ranked_songs and ranked_songs[0][1] > 5:
                        best_song, best_score = ranked_songs[0]
                        st.markdown("<div class='subtitle'>MATCH FOUND</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='match-found'>{best_song}</div>", unsafe_allow_html=True)
                        
                        runner_up_score = ranked_songs[1][1] if len(ranked_songs) > 1 else 1
                        ratio = best_score / max(1, runner_up_score)
                        st.write(f"Cluster score: **{best_score}** | **{ratio:.1f}x** the runner-up")
                        
                        st.markdown("<br><div class='subtitle'>CANDIDATE SCORES</div>", unsafe_allow_html=True)
                        for song, score in ranked_songs[:5]:
                            pct = min(100, int((score / best_score) * 100))
                            st.markdown(f"**{song}** <span style='float:right'>{score}</span>", unsafe_allow_html=True)
                            st.markdown(f"<div class='score-bar-bg'><div class='score-bar-fill' style='width: {pct}%'></div></div><br>", unsafe_allow_html=True)

                        st.divider()
                        st.markdown("**STEP 1 - FEATURE EXTRACTION**")
                        st.markdown("### From spectrogram to constellation")
                        st.write(f"The clip was converted into a time-frequency map. Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and mild noise. **{len(peak_f)} prominent peaks** were kept.")
                        st.pyplot(core.plot_constellation(Sxx, f, t, peak_f, peak_t))

                        st.divider()
                        st.markdown("**STEP 2 - THE PROOF**")
                        st.markdown("### The alignment spike")
                        st.write(f"Every matched hash votes for a time offset. Chance matches scatter randomly, forming a flat noise floor. A genuine match makes them converge. **{best_score} hashes agreed on a single offset.**")
                        st.pyplot(core.plot_alignment_spike(all_matches, best_song))

                    else:
                        st.error("No definitive match found. The clip may be too noisy or not in the library.")

# ----------------- TAB 3: BATCH -----------------
with tab3:
    st.subheader("Identify many clips at once")
    batch_files = st.file_uploader("Upload multiple clips to generate results.csv", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'], accept_multiple_files=True)
    
    if st.button("Run Batch", type="primary"):
        if not db:
             st.error("Database is empty.")
        else:
            results = []
            progress_bar = st.progress(0)
            
            for i, file in enumerate(batch_files):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file.getvalue())
                    tmp_path = tmp.name
                    
                y, sr = core.load_audio(tmp_path)
                f, t, Sxx = core.get_spectrogram(y, sr)
                peak_f, peak_t = core.extract_peaks(Sxx)
                hashes = core.generate_hashes(peak_f, peak_t)
                
                ranked_songs, _ = core.match_query(hashes, db)
                prediction = ranked_songs[0][0] if (ranked_songs and ranked_songs[0][1] > 5) else "NONE"
                
                results.append({"FILE": file.name, "PREDICTION": prediction})
                progress_bar.progress((i + 1) / len(batch_files))
                
            if results:
                st.success(f"{len(results)} clips matched.")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download results.csv", data=csv, file_name="results.csv", mime="text/csv")