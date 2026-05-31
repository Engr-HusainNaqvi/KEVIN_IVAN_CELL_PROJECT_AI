import streamlit as st
import anthropic
import base64
import io
import json
from PIL import Image
import numpy as np

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroCyto Research Planner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  – deep-lab / medical aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
  --bg-deep:   #0a0d14;
  --bg-card:   #111521;
  --bg-panel:  #161b2e;
  --accent1:   #00d4aa;
  --accent2:   #4f8ef7;
  --accent3:   #f76b6b;
  --accent4:   #f0c040;
  --text-main: #e8eaf6;
  --text-muted:#8892b0;
  --border:    #1e2d4a;
  --glow1: 0 0 18px rgba(0,212,170,.25);
  --glow2: 0 0 18px rgba(79,142,247,.25);
}

html, body, .stApp {
  background-color: var(--bg-deep) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text-main);
}

/* ── hide default header ── */
header[data-testid="stHeader"] { background: transparent; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
  background: var(--bg-card) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── main title ── */
.hero-title {
  font-family: 'Instrument Serif', serif;
  font-size: 2.6rem;
  font-style: italic;
  background: linear-gradient(120deg, var(--accent1), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  line-height: 1.2;
  margin-bottom: 0;
}
.hero-sub {
  font-family: 'Space Mono', monospace;
  font-size: .72rem;
  color: var(--text-muted);
  letter-spacing: .12em;
  text-transform: uppercase;
  margin-top: .3rem;
}

/* ── section headers ── */
.section-label {
  font-family: 'Space Mono', monospace;
  font-size: .65rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--accent1);
  border-left: 2px solid var(--accent1);
  padding-left: .6rem;
  margin: 1.6rem 0 .8rem;
}

/* ── cards ── */
.info-card {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.2rem 1.4rem;
  margin-bottom: 1rem;
}
.info-card h4 {
  font-family: 'Space Mono', monospace;
  font-size: .78rem;
  color: var(--accent2);
  margin-bottom: .5rem;
  text-transform: uppercase;
  letter-spacing: .08em;
}
.info-card p { font-size: .88rem; color: var(--text-muted); margin: 0; line-height: 1.6; }

/* ── pipeline step badge ── */
.step-badge {
  display: inline-flex;
  align-items: center;
  gap: .4rem;
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: .35rem .75rem;
  font-family: 'Space Mono', monospace;
  font-size: .7rem;
  color: var(--accent1);
  margin: .25rem .25rem .25rem 0;
}
.step-num {
  background: var(--accent1);
  color: #000;
  border-radius: 4px;
  padding: .1rem .35rem;
  font-weight: 700;
  font-size: .65rem;
}

/* ── AI output container ── */
.ai-output {
  background: var(--bg-panel);
  border: 1px solid var(--accent1);
  border-radius: 12px;
  padding: 1.6rem 1.8rem;
  margin-top: 1rem;
  box-shadow: var(--glow1);
  font-size: .93rem;
  line-height: 1.75;
  color: var(--text-main);
}

/* ── streamlit elements overrides ── */
.stButton > button {
  background: linear-gradient(135deg, var(--accent1), var(--accent2)) !important;
  color: #000 !important;
  font-family: 'Space Mono', monospace !important;
  font-weight: 700 !important;
  font-size: .75rem !important;
  letter-spacing: .08em !important;
  border: none !important;
  border-radius: 8px !important;
  padding: .65rem 1.6rem !important;
  transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }

div[data-testid="stSelectbox"] label,
div[data-testid="stRadio"] label,
div[data-testid="stFileUploader"] label {
  font-family: 'Space Mono', monospace !important;
  font-size: .7rem !important;
  letter-spacing: .1em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}

.stProgress .st-bo { background: var(--accent1) !important; }

/* ── metric tiles ── */
.metric-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: .8rem; margin: 1rem 0; }
.metric-tile {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
  text-align: center;
}
.metric-val {
  font-family: 'Instrument Serif', serif;
  font-size: 1.9rem;
  color: var(--accent1);
  font-style: italic;
}
.metric-lbl {
  font-family: 'Space Mono', monospace;
  font-size: .6rem;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-top: .2rem;
}

/* ── tab styling ── */
button[data-baseweb="tab"] {
  font-family: 'Space Mono', monospace !important;
  font-size: .68rem !important;
  letter-spacing: .1em !important;
  color: var(--text-muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: var(--accent1) !important;
  border-bottom: 2px solid var(--accent1) !important;
}

/* ── scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── code/mono text ── */
code { font-family: 'Space Mono', monospace; font-size: .8rem; color: var(--accent4); }

/* ── divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  KNOWLEDGE BASE  (from the 3 papers)
# ─────────────────────────────────────────────
PAPER_KNOWLEDGE = {
    "Paper 1 – DysplasiaNet (Acevedo et al., 2021)": {
        "task": "Binary classification – hypogranulated vs normal neutrophils (MDS diagnosis)",
        "architecture": "Custom 4-block CNN (DysplasiaNet) — 72,977 trainable params, trained from scratch",
        "dataset": "20,670 cell images (May Grünwald-Giemsa stained, CellaVision DM96, 363×360 px)",
        "training": "ADAM optimizer, binary cross-entropy, 100 epochs, hold-out + 5/10-fold CV",
        "performance": "Sensitivity 95.5% · Specificity 94.3% · Precision 94% · Accuracy 94.85% · AUC 0.982",
        "key_contributions": [
            "First CNN validated specifically for MDS diagnostic support",
            "ROC-based cut-off for granularity score (cell level) + threshold (smear level)",
            "Model interpretability via Grad-CAM activation maps & t-SNE feature reduction",
            "Lightweight design — no GPU required for inference",
            "Training from scratch outperforms fine-tuning for this medical domain",
        ],
        "pipeline_steps": [
            "Image acquisition (CellaVision DM96)",
            "May Grünwald-Giemsa staining normalization",
            "Pathologist annotation (ground truth)",
            "Data augmentation to 3000 images/class",
            "8-model architecture search (vary conv blocks, nodes, FC layers)",
            "2-stage training: 20 epochs screening → 100 epochs fine-train",
            "ROC cut-off optimisation (cell level)",
            "ROC threshold optimisation (smear level)",
            "McNemar test for model comparison",
            "Grad-CAM & feature-map interpretability",
            "Proof-of-concept on held-out patients",
        ],
    },
    "Paper 2 – SNM/GAN Stain Normalisation (Barrera et al., 2023)": {
        "task": "Multi-center stain colour normalisation + 3-class cell classification (ALC, BL, RL)",
        "architecture": "Sequential dual-GAN (PIX2PIX + ResNet34 discriminator) → SENet154 classifier",
        "dataset": "44,822 RC images (Hospital Clínic Barcelona) + 3,141 images from 4 external centres",
        "training": "NoGAN (separate then joint), 50 epochs per GAN; Adam + cyclical LR for classifier",
        "performance": "RC accuracy 95.7% → external centres 92–98.4% after normalisation (vs 61–84% before)",
        "key_contributions": [
            "Two-stage GAN: RGB → adaptive grey (GAN1) → normalised RGB (GAN2)",
            "Morphology preserved while colour distribution mapped to reference centre",
            "FID, IS, LPIPS + cumulative colour histogram (CCH/RMSE) evaluation",
            "Reactive lymphocytes most sensitive to staining variability",
            "Low inference time: ~53 s per 1000 images (SNM) + ~104 s (classification)",
        ],
        "pipeline_steps": [
            "Multi-centre blood smear collection (EDTA, Sysmex SP10, CellaVision DM96)",
            "Establish Reference Centre (RC) staining standard",
            "Build paired RGB ↔ greyscale dataset",
            "Train GAN1 (RGB → adaptive grey) using NoGAN method",
            "Train GAN2 (adaptive grey → normalised RGB) using NoGAN method",
            "Validate normalisation with FID / IS / LPIPS metrics",
            "Cumulative colour histogram (CCH) RMSE analysis",
            "Train SENet154 classifier on RC images only",
            "Test classifier pre- vs post-SNM on external centres",
            "Compare normalisation against CycleGAN / single Pix2Pix baselines",
        ],
    },
    "Paper 3 – NeuNN Cytoplasmic Inclusions (Barrera et al., 2024)": {
        "task": "7-class neutrophil classification (NEU, HYP, CRY, DB, HJBLI, GBI, BAC)",
        "architecture": "EfficientNet-B7 (66M params) trained from scratch; SyntheticCellGAN for rare-class augmentation",
        "dataset": "5,605 original images + GAN-generated GBI class; 50/50 learn/test split",
        "training": "Adam lr=0.001, categorical cross-entropy, early stopping (threshold 0.05, patience 3), 52 epochs",
        "performance": "Overall accuracy 94.3% · Sensitivity 94% · Specificity 99.1% · F1 94.3% · MCC 0.936",
        "key_contributions": [
            "First system for automatic classification of all 6 major neutrophil cytoplasmic inclusions",
            "Two-step GAN pipeline for rare-class (GBI) synthetic image generation with location control",
            "Pix2Pix for cytoplasm-aware inclusion insertion (white-marker → GBI)",
            "MBConv + Squeeze-and-Excitation blocks for channel recalibration",
            "Jaccard Index + MCC used alongside F1 to handle class imbalance",
            "Multi-label coexistence of DB + HYP identified and discussed",
            "Publicly released annotated dataset",
        ],
        "pipeline_steps": [
            "Collect & annotate 7 neutrophil classes (expert pathologists)",
            "Analyse class imbalance — GBI has only 42 images",
            "Apply classical augmentation (rotation, flip, zoom, brightness) for HJBLI/BAC",
            "Train SyntheticCellGAN (WassersteinGAN + SRGAN) for neutrophil base generation",
            "Modify GAN B4 → B4mr: add white-circle cytoplasm markers",
            "Train Pix2Pix GAN-C: convert markers to realistic GBI inclusions",
            "Build hybrid dataset (real + synthetic GBI, 50/50 split)",
            "Balance training set to 200 images/class (sub-sample NEU, upsample others)",
            "Architecture search: ConvNeXt, ResNets, VGGs, ViT-G, EfficientNet-B0→L2",
            "Train EfficientNet-B7 with early stopping",
            "Evaluate with Jaccard Index, MCC, F1, sensitivity, specificity",
            "Softmax probability analysis for ambiguous DB/HYP co-occurrences",
        ],
    },
}

PIPELINE_MASTER = [
    ("🔬", "Image Acquisition",
     "CellaVision DM96 · 360×363 px · RGB · May Grünwald-Giemsa stain"),
    ("🎨", "Stain Normalisation (if multi-centre)",
     "Dual-GAN SNM (Barrera 2023) → FID/IS/LPIPS + CCH-RMSE validation"),
    ("🏷️", "Expert Annotation",
     "Clinical pathologist ground-truth labelling per WHO 2016 classification"),
    ("⚖️", "Class Imbalance Analysis",
     "Check distribution · Subsample majority · Upsample minority · GAN synthesis for rare classes (<50 imgs)"),
    ("🔄", "Data Augmentation",
     "Rotation 0–360° · Vertical/horizontal flip · Zoom 0.5×–2.0× · Brightness ±0.4"),
    ("🏗️", "Architecture Search",
     "Custom lightweight CNN (DysplasiaNet-style) OR EfficientNet-B0→B7 · McNemar test for comparison"),
    ("🎯", "Training Strategy",
     "Adam optimiser · Categorical cross-entropy · Early stopping (patience=3, δ=0.05) · 5/10-fold CV"),
    ("📈", "Performance Evaluation",
     "Sensitivity · Specificity · Precision · F1 · AUC-ROC · Jaccard Index · MCC"),
    ("🔍", "Interpretability",
     "Grad-CAM activation maps · t-SNE feature visualisation · Feature map inspection"),
    ("🧪", "Proof of Concept",
     "Hold-out patient cohort · McNemar pairwise model test · Clinical pathologist review"),
]

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────

def image_to_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_system_prompt() -> str:
    kb = json.dumps(PAPER_KNOWLEDGE, indent=2)
    return f"""You are NeuroCyto, an expert AI research advisor specialising in deep learning for haematological 
blood cell image analysis. You have deep expertise from three landmark papers:

{kb}

When a user uploads a blood cell image, you:
1. Analyse what is visible in the image (cell type, staining quality, morphological features, abnormalities)
2. Recommend a complete, step-by-step ML/image-processing research pipeline tailored to that specific image
3. Cite which paper's methodology is most relevant at each step
4. Flag any challenges (class imbalance, staining variability, rare findings) and how to address them
5. Suggest evaluation metrics appropriate for the task

Format your response with clear numbered sections using these headers:
## 🔬 Image Analysis
## 🎯 Recommended Research Pipeline  
## 🏗️ Architecture Recommendation
## ⚠️ Challenges & Solutions
## 📊 Evaluation Strategy
## 📚 Paper References

Be specific, technical, and actionable. Reference exact methods, hyperparameters, and architectural choices from the papers where relevant."""


def build_user_prompt(task_type: str, centre_count: str, image_description: str) -> str:
    return f"""I have uploaded a peripheral blood cell image. 

Context:
- Research task: {task_type}
- Number of data collection centres: {centre_count}
- Additional notes: {image_description if image_description else "None provided"}

Please analyse this image and provide a complete, tailored ML research pipeline and plan I can present to my professor. 
Include specific architectural choices, training strategies, evaluation metrics, and reference which of the three papers 
(DysplasiaNet, SNM/GAN Normalisation, NeuNN) each step draws from."""


def call_claude_vision(api_key: str, pil_img: Image.Image, task_type: str,
                       centre_count: str, extra_notes: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    img_b64 = image_to_b64(pil_img)

    with st.spinner("🧠 Analysing image and generating research pipeline…"):
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": build_user_prompt(task_type, centre_count, extra_notes),
                        },
                    ],
                }
            ],
        )
    return response.content[0].text


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.5rem 0 1.2rem'>
      <div style='font-family:Space Mono,monospace;font-size:.6rem;letter-spacing:.2em;
                  text-transform:uppercase;color:#4f8ef7;margin-bottom:.4rem'>
        NeuroCyto v1.0
      </div>
      <div style='font-family:Instrument Serif,serif;font-size:1.3rem;font-style:italic;color:#e8eaf6'>
        Blood Cell ML<br/>Research Planner
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">API Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("Anthropic API Key", type="password",
                            placeholder="sk-ant-…",
                            help="Your Anthropic API key — never stored.")

    st.markdown('<div class="section-label">Research Context</div>', unsafe_allow_html=True)
    task_type = st.selectbox(
        "Research Task",
        [
            "MDS / Hypogranulated Neutrophil Detection (Paper 1)",
            "Multi-Centre Stain Normalisation (Paper 2)",
            "Cytoplasmic Inclusion Classification — 7 classes (Paper 3)",
            "Custom / Exploratory Task",
        ],
    )
    centre_count = st.radio("Data Collection Centres", ["Single Centre", "Multi-Centre (2–5)"], horizontal=True)
    extra_notes = st.text_area("Additional Notes (optional)",
                               placeholder="e.g. dataset size, rare class present, target deployment environment…",
                               height=90)

    st.markdown("---")
    st.markdown("""
    <div style='font-family:Space Mono,monospace;font-size:.62rem;color:#8892b0;line-height:1.8'>
      📄 Based on:<br/>
      · Acevedo et al. 2021<br/>
      · Barrera et al. 2023<br/>
      · Barrera et al. 2024<br/><br/>
      🏥 Hospital Clínic Barcelona<br/>
      🔬 UPC BarcelonaTech
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────
st.markdown("""
<div style='padding:1.2rem 0 .5rem'>
  <div class='hero-title'>NeuroCyto Research Planner</div>
  <div class='hero-sub'>Deep Learning · Peripheral Blood Cell Analysis · MDS / Inclusions / Stain Normalisation</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  🖼️  UPLOAD & ANALYSE  ", "  📐  PIPELINE OVERVIEW  ", "  📚  PAPER SUMMARIES  "])

# ───────────── TAB 1 — Upload & Analyse ─────────────
with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Upload Blood Cell Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload Image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            label_visibility="collapsed",
        )

        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption=f"Uploaded: {uploaded.name}", use_container_width=True)

            # Basic image stats
            arr = np.array(pil_img)
            st.markdown('<div class="section-label">Image Statistics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class='metric-grid'>
              <div class='metric-tile'>
                <div class='metric-val'>{pil_img.width}×{pil_img.height}</div>
                <div class='metric-lbl'>Resolution (px)</div>
              </div>
              <div class='metric-tile'>
                <div class='metric-val'>{arr.mean():.1f}</div>
                <div class='metric-lbl'>Mean Intensity</div>
              </div>
              <div class='metric-tile'>
                <div class='metric-val'>{arr.std():.1f}</div>
                <div class='metric-lbl'>Std Deviation</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            r_mean, g_mean, b_mean = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
            st.markdown(f"""
            <div class='info-card'>
              <h4>RGB Channel Analysis</h4>
              <p>
                <span style='color:#f76b6b'>■ R: {r_mean:.1f}</span> &nbsp;
                <span style='color:#00d4aa'>■ G: {g_mean:.1f}</span> &nbsp;
                <span style='color:#4f8ef7'>■ B: {b_mean:.1f}</span>
                <br/><br/>
                Dominant channel: <code>{'RED' if r_mean==max(r_mean,g_mean,b_mean) else 'GREEN' if g_mean==max(r_mean,g_mean,b_mean) else 'BLUE'}</code>
                — useful for stain normalisation baseline assessment.
              </p>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-label">AI Research Pipeline Analysis</div>', unsafe_allow_html=True)

        if not uploaded:
            st.markdown("""
            <div class='info-card' style='text-align:center;padding:3rem 1.4rem'>
              <div style='font-size:3rem;margin-bottom:.8rem'>🔬</div>
              <h4>No Image Uploaded</h4>
              <p>Upload a peripheral blood cell image (neutrophil smear recommended) on the left 
              to receive a tailored ML research pipeline recommendation.</p>
            </div>
            """, unsafe_allow_html=True)
        elif not api_key:
            st.markdown("""
            <div class='info-card' style='border-color:#f0c040'>
              <h4 style='color:#f0c040'>⚠️  API Key Required</h4>
              <p>Enter your Anthropic API key in the sidebar to enable AI-powered image analysis 
              and pipeline generation.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("🚀  Generate Research Pipeline", use_container_width=True):
                try:
                    result = call_claude_vision(api_key, pil_img, task_type, centre_count, extra_notes)
                    st.session_state["last_result"] = result
                except anthropic.AuthenticationError:
                    st.error("❌ Invalid API key. Please check your Anthropic API key in the sidebar.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

            if "last_result" in st.session_state:
                st.markdown(f"""
                <div class='ai-output'>
                {st.session_state['last_result'].replace(chr(10), '<br/>')}
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    "⬇️  Export Pipeline as .txt",
                    data=st.session_state["last_result"],
                    file_name="research_pipeline_plan.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

# ───────────── TAB 2 — Pipeline Overview ─────────────
with tab2:
    st.markdown('<div class="section-label">Master Research Pipeline (from all 3 papers)</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
      <h4>How to use this pipeline</h4>
      <p>Each step below is derived from methods validated across the three papers. 
      Steps 1–4 are universal; step 5 onward depends on your specific classification task and dataset characteristics.
      Upload an image in the <b>Analyse</b> tab to get a customised version of this pipeline for your exact case.</p>
    </div>
    """, unsafe_allow_html=True)

    for i, (icon, title, detail) in enumerate(PIPELINE_MASTER):
        accent = ["#00d4aa","#4f8ef7","#f0c040","#f76b6b","#b47ffc",
                  "#00d4aa","#4f8ef7","#f0c040","#f76b6b","#b47ffc"][i % 10]
        st.markdown(f"""
        <div class='info-card' style='border-left:3px solid {accent};margin-bottom:.7rem'>
          <h4 style='color:{accent}'>{icon} Step {i+1}: {title}</h4>
          <p>{detail}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Key Architectural Choices by Task</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown("""
        <div class='info-card' style='border-top:2px solid #00d4aa'>
          <h4>Binary / MDS Detection</h4>
          <p>DysplasiaNet · 4 conv blocks · 16 filters · 1 FC layer · 72K params<br/><br/>
          ✓ No GPU required<br/>
          ✓ ROC-based granularity score<br/>
          ✓ McNemar comparison<br/>
          ✓ Grad-CAM interpretability</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='info-card' style='border-top:2px solid #4f8ef7'>
          <h4>Multi-Centre Normalisation</h4>
          <p>GAN1: RGB→AdaptiveGrey<br/>GAN2: Grey→NormalisedRGB<br/>
          Classifier: SENet154<br/><br/>
          ✓ NoGAN training (50 epochs)<br/>
          ✓ FID / IS / LPIPS metrics<br/>
          ✓ CCH-RMSE colour analysis<br/>
          ✓ Morphology preserved</p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='info-card' style='border-top:2px solid #f0c040'>
          <h4>7-Class Inclusions</h4>
          <p>EfficientNet-B7 · 66M params<br/>SyntheticCellGAN for rare GBI<br/><br/>
          ✓ 50/50 train/test split<br/>
          ✓ Early stopping (δ=0.05, p=3)<br/>
          ✓ Jaccard + MCC metrics<br/>
          ✓ Multi-label coexistence</p>
        </div>
        """, unsafe_allow_html=True)

    # Hyperparameter reference table
    st.markdown('<div class="section-label">Validated Hyperparameter Reference</div>', unsafe_allow_html=True)
    hparams = {
        "Optimiser": ["Adam", "Adam", "Adam"],
        "Loss Function": ["Binary Cross-Entropy", "GAN loss (PIX2PIX)", "Categorical Cross-Entropy"],
        "Learning Rate": ["Default Adam", "FastAI NoGAN default", "0.001"],
        "Batch Size": ["Standard", "40 images", "20 images"],
        "Epochs": ["100 (+ 20-epoch screen)", "50 (30 sep + 20 joint)", "52 (early stop)"],
        "Augmentation": ["3000 imgs/class (upsample)", "80/20 split + class balance", "Rotate±360 · Zoom 0.5–2× · Brightness ±0.4"],
        "Split Strategy": ["Hold-out + 5/10-fold CV", "80/20 rule", "50/50 (imbalanced dataset)"],
        "Regularisation": ["Dropout p=0.5", "NoGAN stability", "Early stopping"],
    }
    header_row = "| Hyperparameter | Paper 1 (DysplasiaNet) | Paper 2 (SNM-GAN) | Paper 3 (NeuNN) |"
    sep_row    = "|---|---|---|---|"
    rows = [header_row, sep_row]
    for k, vals in hparams.items():
        rows.append(f"| **{k}** | {vals[0]} | {vals[1]} | {vals[2]} |")
    st.markdown("\n".join(rows))


# ───────────── TAB 3 — Paper Summaries ─────────────
with tab3:
    st.markdown('<div class="section-label">Literature Foundation</div>', unsafe_allow_html=True)

    for paper_name, details in PAPER_KNOWLEDGE.items():
        accent = "#00d4aa" if "Paper 1" in paper_name else ("#4f8ef7" if "Paper 2" in paper_name else "#f0c040")

        with st.expander(f"  {paper_name}", expanded=False):
            c1, c2 = st.columns([1, 1], gap="medium")
            with c1:
                st.markdown(f"""
                <div class='info-card' style='border-left:3px solid {accent}'>
                  <h4>Task</h4>
                  <p>{details['task']}</p>
                </div>
                <div class='info-card' style='border-left:3px solid {accent}'>
                  <h4>Architecture</h4>
                  <p>{details['architecture']}</p>
                </div>
                <div class='info-card' style='border-left:3px solid {accent}'>
                  <h4>Dataset</h4>
                  <p>{details['dataset']}</p>
                </div>
                <div class='info-card' style='border-left:3px solid {accent}'>
                  <h4>Performance</h4>
                  <p>{details['performance']}</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='info-card' style='border-left:3px solid {accent}'><h4>Key Contributions</h4>", unsafe_allow_html=True)
                for kc in details["key_contributions"]:
                    st.markdown(f"- {kc}")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(f"<div class='info-card' style='border-left:3px solid {accent}'><h4>Pipeline Steps</h4>", unsafe_allow_html=True)
                for idx, step in enumerate(details["pipeline_steps"], 1):
                    st.markdown(f'<span class="step-badge"><span class="step-num">{idx}</span>{step}</span>',
                                unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # Bottom reference section
    st.markdown("---")
    st.markdown('<div class="section-label">Full Paper Citations</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class='info-card'>
      <h4>References</h4>
      <p>
        <b>[1]</b> Acevedo A, Merino A, Boldú L, Molina Á, Alférez S, Rodellar J. 
        <i>A new convolutional neural network predictive model for the automatic recognition of hypogranulated 
        neutrophils in myelodysplastic syndromes.</i> 
        Computers in Biology and Medicine, 134, 104479. (2021)
        <br/><br/>
        <b>[2]</b> Barrera K, Rodellar J, Alférez S, Merino A. 
        <i>Automatic normalized digital color staining in the recognition of abnormal blood cells using 
        generative adversarial networks.</i> 
        Computer Methods and Programs in Biomedicine, 240, 107629. (2023)
        <br/><br/>
        <b>[3]</b> Barrera K, Rodellar J, Alférez S, Merino A. 
        <i>A deep learning approach for automatic recognition of abnormalities in the cytoplasm of neutrophils.</i> 
        Computers in Biology and Medicine, 178, 108691. (2024)
      </p>
    </div>
    """, unsafe_allow_html=True)
