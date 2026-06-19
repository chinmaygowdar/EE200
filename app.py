import streamlit as st
import pandas as pd
import tempfile
import pickle
import os
import core

st.set_page_config(page_title="EE200: Audio Fingerprinting", layout="wide")
st.title("🎧 EE200: Audio Fingerprinting")
st.markdown("Signals, Systems & Networks - Project Demo")

# Load Database
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
    st.header("Library Index")
    if not db:
        st.error("Database not found or is empty. Please run `build_index.py` to generate `database.pkl`.")
    else:
        songs = set(s for hashes in db.values() for s, _ in hashes)
        st.metric(label="Total Unique Hashes", value=f"{len(db):,}")
        st.metric(label="Total Indexed Songs", value=len(songs))
        st.write("**Indexed Tracks:**", list(songs))

# ----------------- TAB 2: IDENTIFY -----------------
with tab2:
    st.header("Identify a Clip")
    uploaded_file = st.file_uploader("Upload audio clip", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'])
    
    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        if st.button("Identify Song"):
            if not db:
                st.error("Cannot identify: Database is empty. Go index the library first.")
            else:
                with st.spinner("Extracting constellation & hashing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                        
                    # Signal Processing Pipeline
                    y, sr = core.load_audio(tmp_path)
                    f, t, Sxx = core.get_spectrogram(y, sr)
                    peak_f, peak_t = core.extract_peaks(Sxx)
                    hashes = core.generate_hashes(peak_f, peak_t)
                    
                    # Always show Step 1 (Constellation) so user can see if audio has data
                    st.divider()
                    st.subheader("Step 1: Spectrogram to Constellation")
                    st.markdown(f"Extracted {len(peak_f)} prominent peaks and generated {len(hashes)} hashes.")
                    st.pyplot(core.plot_constellation(Sxx, f, t, peak_f, peak_t))
                    
                    # Search Database
                    best_song, best_score, best_offsets = core.match_query(hashes, db)
                    
                    if best_song and best_score > 5:
                        st.success(f"### MATCH FOUND: **{best_song}**")
                        st.write(f"Alignment Score: {best_score} matching hashes")
                        
                        st.divider()
                        st.subheader("Step 2: The Alignment Spike")
                        st.markdown("A genuine match makes the hashes converge and agree on a single time offset.")
                        st.pyplot(core.plot_offset_histogram(best_offsets))
                    else:
                        st.error("No definitive match found in the database. The audio might not be in the library, or the clip is too noisy/distorted.")

# ----------------- TAB 3: BATCH -----------------
with tab3:
    st.header("Batch Identification")
    st.markdown("Upload multiple query clips to generate a standard `results.csv`.")
    batch_files = st.file_uploader("Upload multiple clips", type=['wav', 'mp3', 'flac', 'ogg', 'm4a'], accept_multiple_files=True)
    
    if st.button("Run Batch"):
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
                
                best_song, best_score, _ = core.match_query(hashes, db)
                
                prediction = best_song if (best_song and best_score > 5) else "NONE"
                results.append({"filename": file.name, "prediction": prediction})
                
                progress_bar.progress((i + 1) / len(batch_files))
                
            if results:
                df = pd.DataFrame(results)
                st.dataframe(df)
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download results.csv", data=csv, file_name="results.csv", mime="text/csv")