import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import os
from sgfmill import sgf

# Import local modules
try:
    from src.predict import get_best_moves, predict_opponent_response, SimpleGameState, load_model
    from src.xai import generate_heatmap
    from src.heuristics import get_smart_explanation
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# Vision module check
try:
    from src.vision import process_image_to_board
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

#  ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Beyond The Move", layout="wide", page_icon="assets/favicon.ico")

# ---------------------- CSS STYLING ----------------------
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root { --bg: #040404; --bg2: #0A0F0E; --panel: #1C1F1E; --text: #F9F9F9; --accent: #0EA66B; --accent-glow: #94FFD9; --silver: #BDC3C7; }
html, body, [data-testid="stAppViewContainer"] { background-color: var(--bg); color: var(--text); font-family: 'Space Grotesk', sans-serif; }
h1, h2, h3, h4, h5, h6 { font-family: 'Cormorant Garamond', serif !important; color: var(--accent-glow) !important; letter-spacing: 0.5px; }
[data-testid="stSidebar"] { background-color: var(--bg2) !important; border-right: 1px solid var(--silver); }
.stButton > button { background-color: var(--accent); color: black; border-radius: 8px; font-weight: 600; border: none; }
.stButton > button:hover { background-color: var(--accent-glow); transform: scale(1.03); }
/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { background-color: var(--panel); border-radius: 5px; color: var(--silver); }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: var(--accent); color: black; }
</style>
""", unsafe_allow_html=True)

# ---------------------- HELPER FUNCTIONS ----------------------
def reset_state():
    st.session_state.analysis_done = False
    st.session_state.current_board = None
    st.session_state.analysis_results = None

if "analysis_done" not in st.session_state: st.session_state.analysis_done = False
if "current_board" not in st.session_state: st.session_state.current_board = None

def draw_board(board_matrix, suggestion=None, heatmap=None):
    if board_matrix is None: return None
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_facecolor('#DCB35C')
    
    # 1. Grid
    for i in range(19):
        ax.plot([i, i], [0, 18], 'k-', lw=0.6, alpha=0.7)
        ax.plot([0, 18], [i, i], 'k-', lw=0.6, alpha=0.7)
    
    # 2. Stars
    stars = [(3,3), (3,9), (3,15), (9,3), (9,9), (9,15), (15,3), (15,9), (15,15)]
    for sx, sy in stars: ax.plot(sx, sy, 'ko', markersize=4)

    # 3. Coordinates
    letters = "ABCDEFGHJKLMNOPQRST"
    for i in range(19):
        # X Labels (A-T)
        ax.text(i, -1.2, letters[i], ha='center', va='center', fontsize=9, fontweight='bold', color='#333')
        # Y Labels (1-19) - 0 is bottom (1), 18 is top (19)
        ax.text(-1.2, i, str(i + 1), ha='center', va='center', fontsize=9, fontweight='bold', color='#333')

    # 4. Heatmap
    if heatmap is not None and heatmap.shape == (19, 19):
        masked_heatmap = np.ma.masked_where(heatmap < 0.15, heatmap)
        ax.imshow(masked_heatmap, extent=[-0.5, 18.5, -0.5, 18.5], origin='lower', cmap='RdYlGn', vmin=0, vmax=1, alpha=0.6, zorder=2)

    # 5. Stones
    for r in range(19):
        for c in range(19):
            if board_matrix[r][c] == 0: continue
            
            x, y = c, r
            
            if board_matrix[r][c] == 1: # Black
                ax.add_artist(plt.Circle((x, y), 0.45, color='black', ec='black', zorder=3))
                ax.add_artist(plt.Circle((x-0.15, y+0.15), 0.1, color='white', alpha=0.1, zorder=3.5)) # Shine
            elif board_matrix[r][c] == 2: # White
                ax.add_artist(plt.Circle((x, y), 0.45, color='white', ec='#333', lw=0.5, zorder=3))
                ax.add_artist(plt.Circle((x+0.1, y-0.1), 0.45, color='gray', alpha=0.3, zorder=2.5)) # Shadow

    # 6. Suggestion
    if suggestion:
        r, c = suggestion
        rect = plt.Rectangle((c-0.5, r-0.5), 1, 1, linewidth=2.5, edgecolor='#00FF00', facecolor='none', zorder=4)
        ax.add_patch(rect)
        ax.text(c, r, "AI", color='#00FF00', ha='center', va='center', fontsize=8, fontweight='bold', zorder=5)

    ax.set_xlim(-1.5, 19.5)
    ax.set_ylim(-1.5, 19.5)
    ax.axis('off')
    return fig

def robust_decode(file_bytes):
    """ Tries multiple encodings to decode SGF data. """
    encodings = ['utf-8', 'gb18030', 'latin-1']
    for enc in encodings:
        try:
            return file_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return None

def compute_dataset_stats():
    """Reads a subset of processed files to generate a move heatmap."""
    files = glob.glob("data/processed/*.npz")
    if not files: return None
    
    # Sample up to 100 files to keep it fast
    sample_files = files[:100] 
    global_heatmap = np.zeros((19, 19))
    
    for f in sample_files:
        try:
            data = np.load(f)
            targets = data['targets'] # 1D array of move indices (0-360)
            for t in targets:
                r, c = t // 19, t % 19
                global_heatmap[r][c] += 1
        except: pass
        
    return global_heatmap

# ---------------------- SIDEBAR ----------------------
if os.path.exists("assets/logo.png"):
    st.sidebar.image("assets/logo.png", use_container_width=True)

st.sidebar.title("Beyond the Move")
page = st.sidebar.radio("Navigation", ["1. Home", "2. Interactive Analysis", "3. Model Metrics", "4. Roadmap & Researcher"])

st.sidebar.divider()
rank_mode = st.sidebar.select_slider("Explanation Level", ["Novice (18k-10k)", "Intermediate (9k-1d)", "Expert (2d+)"])
player_color = st.sidebar.radio("I am playing as:", ["Black", "White"])

# ---------------------- PAGE 1: HOME ----------------------
if page == "1. Home":
    st.title("Beyond the Move")
    st.subheader("An Explainable AI (XAI) Framework for Cognitive Skill Acquisition in Go")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("""
        **The Problem: The 'Black Box' Dilemma**\n\n"
                "Modern AI engines (like AlphaGo) act as 'Oracles'—they provide "
                "superhuman move recommendations but offer no explanation for *why* "
                "a move is good. This leaves learners with correct answers but "
                "zero understanding.\n\n"
        **The Solution: A 'Glass Box' Framework**\n\n"
                   "This project bridges the gap by combining three technologies:\n"
                   "* **Computer Vision:** To see the board.\n"
                   "* **Deep Learning (ResNet-18):** To find the moves.\n"
                   "* **Explainable AI (XAI):** To visualize the hidden logic using "
                   "heatmaps and heuristics.
        """)
        st.info(" Select **'Interactive Analysis'** to start using the tool.")
    
    with c2:
        if os.path.exists("assets/ai_of_go.png"):
            st.image("assets/ai_of_go.png", caption="Visualizing Intuition")
    st.divider()

    st.markdown("###  How to Use the System")

    with st.expander("Click here for Instructions", expanded=True):
        st.markdown("""
        **1. Select Input Method:**
        * 📂 **Upload SGF:** Load a digital game record to analyze historical matches.
        * 📷 **Computer Vision (Experimental):** Upload a photo of a physical board.
        
        **2. Configure Analysis Parameters:**
        * 🏳️ **Perspective:** Choose 'Black' or 'White' to orient the AI's advice.
        * 🧠 **Expertise Level:** Select (Novice/Intermediate/Expert) to adjust the explanation complexity.
        
        **3. Generate Insight:**
        * Click **'Analyze Position'**.
        * Watch for the **Heatmap Overlay**:
            * 🟢 **Green Zones:** Safe territory / Good influence.
            * 🔴 **Red Zones:** Urgent threats / Tactical weaknesses.
            
        **4. Review the Explanation:**
        * Read the generated text heuristic (e.g., *"The AI is protecting the corner group"*).
        * Compare the heatmaps to understand the *reasoning* behind the move.
        """)
# ---------------------- PAGE 2: INTERACTIVE ANALYSIS ----------------------
elif page == "2. Interactive Analysis":
    st.header("AI Game Analysis")
    
    if not AI_AVAILABLE:
        st.warning("⚠️ Predictive AI Modules not found! (src/predict.py missing). You can still test Vision.")
    
    # Input Method Tabs
    tab_sgf, tab_img = st.tabs(["📂 Upload SGF", "📷 Upload Photo(WIP)"])
    
    board_loaded = False

    # --- TAB 1: SGF ---
    with tab_sgf:
        uploaded_sgf = st.file_uploader("Choose SGF File", type=['sgf'], key="sgf_uploader", on_change=reset_state)
        if uploaded_sgf:
            try:
                # 1. READ BYTES
                file_bytes = uploaded_sgf.read()
                
                # 2. ROBUST DECODE
                sgf_content = robust_decode(file_bytes)
                
                if sgf_content is None:
                    st.error("❌ Failed to decode SGF file. Unknown encoding.")
                else:
                    # 3. PARSE
                    try:
                        game = sgf.Sgf_game.from_string(sgf_content)
                        board_arr = np.zeros((19,19), dtype=int)
                        
                        # Replay moves
                        for node in game.get_main_sequence():
                            color, move = node.get_move()
                            if move:
                                r, c = move
                                board_arr[r][c] = 1 if color == 'b' else 2
                        
                        st.session_state.current_board = board_arr
                        board_loaded = True
                        st.success("✅ Game loaded successfully!")
                        
                    except ValueError as e:
                        st.error(f"❌ SGF Syntax Error: {e}")
                    except Exception as e:
                        st.error(f"❌ Error processing game: {e}")

            except Exception as e:
                st.error(f"❌ Critical Error: {e}")

    # --- TAB 2: IMAGE ---
    with tab_img:
        st.warning("🚧 **Work in Progress**: The Computer Vision module is currently experimental. Stone detection results inaccurate.")
        st.info("For testing purposes only. Please upload clear, top-down images.")

        img_file = st.file_uploader("Upload Board Image", type=['png', 'jpg', 'jpeg'], key="img_uploader")
        
        if img_file:
            # Layout: Input on Left, Debug on Right
            c1, c2 = st.columns(2)
            with c1:
                st.image(img_file, caption="1. Input Image", use_container_width=True)
            
            if st.button("Process Image"):
                if VISION_AVAILABLE:
                    with st.spinner("Scanning board..."):
                        # Save temp file
                        temp_filename = "temp_board.jpg"
                        with open(temp_filename, "wb") as f:
                            f.write(img_file.getbuffer())
                        
                        try:
                            # 1. Process
                            # Robustly handle both old (1 return) and new (2 returns) vision modules
                            vision_result = process_image_to_board(temp_filename)
                            
                            debug_img = None
                            if isinstance(vision_result, tuple) and len(vision_result) == 2:
                                board_arr, debug_img = vision_result
                            else:
                                board_arr = vision_result
                            
                            # 2. Show Debug Image
                            with c2:
                                if debug_img is not None:
                                    # Convert BGR (OpenCV) to RGB (Streamlit) using Numpy slicing
                                    # (Avoids needing 'import cv2' in the main app file)
                                    debug_rgb = debug_img[:, :, ::-1] 
                                    st.image(debug_rgb, caption="2. Computer Vision Debug View", use_container_width=True)
                                    st.caption("Check if the green grid aligns with the intersections!")
                                else:
                                    st.warning("Debug view unavailable. Update src/vision.py to see the grid.")

                            # 3. Set State
                            st.session_state.current_board = board_arr
                            st.session_state.analysis_done = False # Reset analysis
                            st.success("Board processed!")
                            
                            # Cleanup
                            if os.path.exists(temp_filename):
                                os.remove(temp_filename)
                                
                        except Exception as e:
                            st.error(f"Vision Error: {e}")
                else:
                    st.error("Vision module not found.")

    # --- ANALYSIS LOGIC ---
    if st.session_state.current_board is not None:
        c1, c2 = st.columns([1.3, 1])
        
        with c2:
            st.subheader("Analysis Control")
            if st.button("🧠 Analyze Position", type="primary"):
                if not AI_AVAILABLE:
                    st.error("AI modules missing. Cannot analyze.")
                else:
                    with st.spinner("Consulting AI Strategist..."):
                        # 1. Prediction
                        stone_count = np.count_nonzero(st.session_state.current_board)
                        st.write(f"DEBUG: Board has {stone_count} stones.")
                        
                        try:
                            moves = get_best_moves(st.session_state.current_board, player_color)
                            
                            if moves:
                                best_move = moves[0]
                                
                                # 2. XAI Heatmap
                                model = load_model()
                                if model:
                                    game_state = SimpleGameState(st.session_state.current_board, 'b' if player_color=='Black' else 'w')
                                    heatmap, sig_stones = generate_heatmap(model, game_state, player_color, (best_move['row'], best_move['col']))
                                    
                                    # 3. Rival Response
                                    rival_move = predict_opponent_response(st.session_state.current_board, best_move['row'], best_move['col'], player_color)
                                    
                                    # 4. Smart Text
                                    text = get_smart_explanation(best_move, rank_mode, sig_stones, rival_move)
                                    
                                    st.session_state.analysis_results = {
                                        "top_moves": moves,
                                        "heatmap": heatmap,
                                        "text": text,
                                        "rival": rival_move
                                    }
                                    st.session_state.analysis_done = True
                                else:
                                    st.error("Model failed to load. Check 'models/' folder.")
                            else:
                                st.warning("AI was silent. Ensure model is trained.")
                        except Exception as e:
                            st.error(f"AI Analysis failed: {e}")

            # Display Results
            if st.session_state.analysis_done:
                res = st.session_state.analysis_results
                
                st.markdown("### 🏆 Top Recommendations")
                df_moves = pd.DataFrame(res['top_moves'])
                
                # --- FIX: KeyError 'rank' ---
                df_moves['rank'] = range(1, len(df_moves) + 1)
                
                if 'confidence' in df_moves.columns:
                    df_moves = df_moves[['rank', 'move', 'confidence']]
                else:
                    df_moves = df_moves[['rank', 'move']]
                
                st.dataframe(df_moves.set_index('rank'), use_container_width=True)
                
                st.markdown("### 📝 Strategic Insight")
                st.info(res['text'])

        with c1:
            st.subheader("Visual Board")
            
            heatmap = None
            suggestion = None
            
            if st.session_state.analysis_done:
                heatmap = st.session_state.analysis_results.get('heatmap')
                top_moves = st.session_state.analysis_results.get('top_moves')
                if top_moves:
                    suggestion = (top_moves[0]['row'], top_moves[0]['col'])
            
            fig = draw_board(st.session_state.current_board, suggestion, heatmap)
            if fig:
                st.pyplot(fig)

# ---------------------- PAGE 3: METRICS ----------------------
elif page == "3. Model Metrics":
    st.header("📊 System Performance")
    
    # Create Tabs for Training and Data Analysis
    tab1, tab2 = st.tabs(["📉 Training & Validation", "🔍 Data Analysis"])

    # TAB 1: TRAINING METRICS
    with tab1:
        st.subheader("Model Convergence")
        if os.path.exists("assets/model_stats.json"):
            with open("assets/model_stats.json") as f: 
                stats = json.load(f)
            
            # Display 4 Metrics side-by-side
            c1, c2, c3 = st.columns(3)
            c1.metric("Games Learned", stats.get('total_games_available', 0))
            t_loss = stats.get('final_train_loss', 0)
            
            c2.metric("Train Loss", f"{t_loss:.4f}")

            c3.metric("Training Time", f"{stats.get('training_time_min', 0)} min")
        else:
            st.warning("No training stats found. Please run src/train.py first.")

     # TAB 2: DATA ANALYSIS
    with tab2:
        st.subheader("Raw Data Distribution (Fuseki)")
        st.markdown("This heatmap visualizes the **Global Move Probability** derived from the raw professional dataset (FoxGo). It answers: *Where do Pros play?*")
        
        if st.button("Generate Dataset Heatmap"):
            with st.spinner("Scanning processed tensors..."):
                # Reuse the function defined at the top of app.py
                global_hm = compute_dataset_stats()
                
            if global_hm is not None:
                fig, ax = plt.subplots(figsize=(6,6))
                # Use a heat-color map
                sns.heatmap(global_hm, cmap="inferno", square=True, cbar=True, ax=ax, xticklabels=False, yticklabels=False)
                ax.invert_yaxis() # Go board row 1 is at bottom
                ax.set_title("Professional Move Distribution (First 100 Games Sample)")
                st.pyplot(fig)
                st.caption("Brighter areas indicate higher move frequency (Corners are most popular).")
            else:
                st.error("No processed data found in 'data/processed/'. Run src/process.py first.")
# ---------------------- PAGE 4: ROADMAP ----------------------
elif page == "4. Roadmap & Researcher":
    st.title("📍 Project Roadmap & Researcher Profile")
    st.markdown("---")

    col1, col2 = st.columns([2, 1], gap="large")

    # --- LEFT COLUMN: ROADMAP ---
    with col1:
        st.subheader("🚀 Project Development Roadmap")
        st.markdown(
            "This research is structured to transition from a technical baseline (Phase I) "
            "to advanced semantic explainability and human-centric evaluation (Phase II)."
        )
        
        # --- Phase I (Done) ---
        with st.container():
            st.success("### ✅ Phase I: Baseline Architecture (Completed)")
            st.markdown("""
            **Focus:** Establishing the 'Glass Box' Core.
            * **Deep Learning:** Implemented ResNet-18 with a custom $19 \\times 19 \\times 17$ tensor encoding pipeline.
            * **XAI Engine:** Integrated 'Captum' library to generate pixel-level attributions via **Integrated Gradients**.
            * **Deployment:** Developed this Streamlit interface for interactive analysis.
            """)
            st.progress(1.0) # 100% complete
            st.caption("*Status: Delivered for Graduation Project I (Jan 2026)*")

        st.markdown("### ⬇️")

        # --- Phase II (Planned) ---
        with st.container():
            st.info("### 🛠️ Phase II: Semantic Explainability (Feb - June 2026)")
            st.markdown("""
            **Focus:** From "Where" to "Why" (Graduation Project II).
            * **Advanced XAI:** Integration of **Grad-CAM** and **Saliency Maps** to visualize high-level shape recognition (e.g., 'Tiger's Mouth').
            * **Comparative Study:** Benchmarking XAI fidelity (Integrated Gradients vs. Grad-CAM) against human professional intuition.
            * **EU Alignment:** Research activities aligned with **COST Action CA22145 (GameTable)**, specifically Working Group 1 (Explainability).
            """)
            st.progress(0.15) # 15% complete (Planning phase)

    # --- RIGHT COLUMN: RESEARCHER PROFILE ---
    with col2:
        st.markdown("### 👩‍💻 Researcher Profile")
        
        # Profile Image with fallback
        if os.path.exists("assets/researcher_avatar.png"):
            st.image("assets/researcher_avatar.png", use_container_width=True)
        else:
            st.warning("Avatar not found. Add 'researcher_avatar.png' to assets/")
            st.markdown("🤖") 

        # Student Details
        st.markdown("### **İpek Naz Sipahi**")
        st.markdown("**B.E. Candidate, Computer Engineering**")
        st.markdown("*Manisa Celal Bayar University*")
        
        st.markdown("---")

        # --- NEW: ADVISOR SECTION ---
        st.markdown("**🎓 Project Advisor**")
        st.markdown("**Dr. Gamze Türkmen**")
        st.caption("Department of Computer Engineering")
        
        st.markdown("---")
        
        # Research Focus
        st.markdown("**🔬 Research Focus**")
        st.caption("Explainable AI (XAI) • Cognitive Science • Game Theory")
        
        st.markdown("**🇪🇺 Memberships**")
        st.caption("Member, COST Action CA22145 (GameTable) - WG1")

        st.markdown("---")
        
        # Contact Links
        st.markdown(
            """
            <div style="display: flex; justify-content: space-around;">
                <a href="https://github.com/4idoneus" target="_blank" style="text-decoration: none;">
                    <button style="background-color:#24292e; color:white; border:none; padding:8px 16px; border-radius:5px; cursor:pointer;">
                        GitHub
                    </button>
                </a>
                <a href="https://www.linkedin.com/in/ipeknazsipahi/" target="_blank" style="text-decoration: none;">
                    <button style="background-color:#0077b5; color:white; border:none; padding:8px 16px; border-radius:5px; cursor:pointer;">
                        LinkedIn
                    </button>
                </a>
            </div>
            """, 
            unsafe_allow_html=True
        )