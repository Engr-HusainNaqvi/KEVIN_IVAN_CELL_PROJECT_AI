# 🔬 NeuroCyto — Blood Cell ML Research Planner

A Streamlit app that analyses peripheral blood cell images and generates a complete, 
paper-backed ML research pipeline for your professor presentation.

---

## What it does

Upload any peripheral blood cell image (neutrophil smear, etc.) and the app will:

1. **Analyse** the image (cell morphology, staining quality, RGB channels)
2. **Generate** a step-by-step ML research pipeline tailored to that image
3. **Cite** exact methods from 3 landmark papers at each step
4. **Flag** challenges (class imbalance, stain variability, rare findings) with solutions
5. **Export** the full plan as a `.txt` file to share with your professor

---

## Based on these papers

| # | Paper | Task | Best Accuracy |
|---|-------|------|---------------|
| 1 | Acevedo et al. 2021 — *DysplasiaNet* | MDS / Hypogranulated neutrophils | 94.85% |
| 2 | Barrera et al. 2023 — *SNM-GAN* | Multi-centre stain normalisation | 98.4% (post-norm) |
| 3 | Barrera et al. 2024 — *NeuNN* | 7-class cytoplasmic inclusions | 94.3% |

---

## Quick Start (local)

```bash
git clone https://github.com/YOUR_USERNAME/neurocyto-research-planner
cd neurocyto-research-planner
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Deploy on Streamlit Cloud

1. Fork / push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, branch `main`, file `app.py`
4. Click **Deploy**
5. Add your `ANTHROPIC_API_KEY` in **Secrets** (optional — users can enter their own key in the sidebar)

---

## Usage

1. Enter your **Anthropic API key** in the sidebar  
2. Select your **Research Task** (MDS, stain normalisation, or 7-class inclusions)  
3. Choose **single or multi-centre** data collection  
4. Upload your **blood cell image**  
5. Click **Generate Research Pipeline**  
6. Download the result with the **Export** button

---

## Files

```
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Dark-theme Streamlit config
└── README.md
```

---

## Requirements

- Python ≥ 3.9  
- Anthropic API key (claude-opus-4-5 vision)
