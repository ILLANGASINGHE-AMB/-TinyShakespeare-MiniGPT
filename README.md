# 🎭 TinyShakespeare — MiniGPT

> A **3.22-million-parameter GPT-style decoder-only transformer** trained from scratch on the Tiny Shakespeare dataset.  
> Built step-by-step through exploration, tokenization, pre-training, evaluation, and fine-tuning — fully documented in Jupyter Notebooks.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Model Architecture](#-model-architecture)
- [Learning Path](#-learning-path)
- [Project Structure](#-project-structure)
- [Notebooks](#-notebooks)
- [Results & Analysis](#-results--analysis)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)

---

## 🧠 Overview

This project builds a **MiniGPT** — a compact, decoder-only transformer — trained entirely on Shakespeare's works using the [Tiny Shakespeare dataset](https://www.kaggle.com/datasets/tunguz/tiny-shakespeare).

The model was built from scratch using **PyTorch**, with no pre-trained weights or external LLM libraries. The full pipeline covers everything from raw text ingestion to character-level text generation, fine-tuning analysis, and final performance evaluation.

**Key highlights:**
- ✅ 3.22 million learnable parameters
- ✅ 4 stacked Transformer decoder blocks
- ✅ Character-level tokenizer (63 unique tokens)
- ✅ Pre-training + Fine-tuning pipeline
- ✅ Text generation & evaluation
- ✅ Baseline vs. fine-tuned comparison

---

## 🏗️ Model Architecture

MiniGPT is a **decoder-only transformer** (GPT-style). The forward pass flows as follows:

```
              MiniGPT
                 │
                 ▼
           Token Embedding
                 +
          Position Embedding
                 │
                 ▼
       ┌───────────────────┐
       │ Transformer Block │  ×4
       ├───────────────────┤
       │  Self-Attention   │
       │  Feed Forward     │
       │  LayerNorm        │
       │  Residuals        │
       └───────────────────┘
                 │
                 ▼
            LayerNorm
                 │
                 ▼
             LM Head
                 │
                 ▼
          63 logits (vocab)
```

### Architecture Details

| Component | Details |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Total Parameters | ~3.22 Million |
| Transformer Blocks | 4 stacked blocks |
| Tokenization | Character-level |
| Vocabulary Size | 63 unique characters |
| Output Shape | `[batch, sequence, 63]` |
| Embeddings | Token Embedding + Positional Embedding |
| Normalization | LayerNorm (pre-norm style) |
| Residual Connections | ✅ Yes |

---

## 🗺️ Learning Path

The project follows a structured, end-to-end ML pipeline:

```
Kaggle Dataset
      ↓
Download / Kaggle API
      ↓
Jupyter Notebook
      ↓
Exploration
      ↓
Data Cleaning
      ↓
Deduplication
      ↓
Train / Validation / Test Split
      ↓
Tokenizer
      ↓
Dataset Preparation
      ↓
Small Transformer / GPT (MiniGPT)
      ↓
Pre-training
      ↓
Evaluation
      ↓
Fine-tuning
      ↓
Text Generation
```

---

## 📁 Project Structure

```
TinyShakespeare-MiniGPT/
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb     # EDA on the Tiny Shakespeare corpus
│   ├── 02_data_cleaning.ipynb           # Cleaning & deduplication
│   ├── 03_tokenization.ipynb            # Character-level tokenizer
│   ├── 04_dataset_preparation.ipynb     # Train/Val/Test split & batching
│   ├── 05_build_transformer.ipynb       # MiniGPT architecture definition
│   ├── 06_train_model.ipynb             # Pre-training loop
│   ├── 07_generate_text.ipynb           # Text generation & sampling
│   ├── 08_evaluation.ipynb              # Model evaluation & metrics
│   ├── 09_fine_tuning.ipynb             # Fine-tuning on specific styles
│   └── 10_final_analysis.ipynb          # Full comparative analysis
│
├── data/
│   ├── raw/
│   │   └── tiny-shakespeare.txt         # Raw dataset (~1.1MB)
│   └── processed/                       # Tokenized & batched data
│
├── models/
│   ├── minigpt_checkpoint.pt            # Pre-trained model weights (~38MB)
│   ├── minigpt_finetuned_checkpoint.pt  # Fine-tuned model weights (~38MB)
│   └── baseline_evaluation.json        # Baseline evaluation metrics
│
├── 3.22-million-parameter decoder.md   # Model architecture diagram
├── LLM_Structure.md                    # High-level LLM structure
├── learning_path.md                    # Training pipeline flowchart
├── project_structure.md                # Project layout reference
└── README.md
```

---

## 📒 Notebooks

| # | Notebook | Description |
|---|---|---|
| 01 | `01_dataset_exploration.ipynb` | Loads and explores the Tiny Shakespeare corpus — character distribution, sequence stats |
| 02 | `02_data_cleaning.ipynb` | Cleans raw text, removes noise and duplicates |
| 03 | `03_tokenization.ipynb` | Builds a character-level tokenizer with encode/decode functions |
| 04 | `04_dataset_preparation.ipynb` | Creates train/val/test splits and batching utilities |
| 05 | `05_build_transformer.ipynb` | Defines the full MiniGPT model in PyTorch (3.22M params) |
| 06 | `06_train_model.ipynb` | Runs the pre-training loop with loss tracking |
| 07 | `07_generate_text.ipynb` | Generates Shakespeare-style text using greedy/temperature sampling |
| 08 | `08_evaluation.ipynb` | Evaluates model loss, perplexity, and generation quality |
| 09 | `09_fine_tuning.ipynb` | Fine-tunes the pre-trained model on a specific style subset |
| 10 | `10_final_analysis.ipynb` | Full comparative analysis: baseline vs. fine-tuned model |

---

## 📊 Results & Analysis

The project includes visual comparisons between the **baseline** pre-trained model and the **fine-tuned** model:

**Generated Analysis Charts (in `notebooks/`):**
- `minigpt_baseline_vs_finetuned.png` — Side-by-side loss curves
- `minigpt_finetuning_curve.png` — Fine-tuning training progression
- `minigpt_finetuning_impact.png` — Fine-tuning impact analysis
- `minigpt_linkedin_analysis.png` — Comprehensive performance summary

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone git@github.com:ILLANGASINGHE-AMB/-TinyShakespeare-MiniGPT.git
cd -TinyShakespeare-MiniGPT
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch Jupyter
```bash
jupyter notebook
```

Then run the notebooks in order from `01` to `10`.

### 5. (Optional) Download the dataset via Kaggle API
```bash
kaggle datasets download -d tunguz/tiny-shakespeare
```

---

## 📦 Requirements

Key dependencies (see `requirements.txt` for exact versions):

| Package | Purpose |
|---|---|
| `torch` | PyTorch — model training & inference |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting training curves |
| `jupyter` | Interactive notebook environment |
| `kaggle` | Dataset download via Kaggle API |

---

## 👤 Author

**Anjana Illangasinghe**  
GitHub: [@ILLANGASINGHE-AMB](https://github.com/ILLANGASINGHE-AMB)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> *"All the world's a stage, and all the men and women merely players."*  
> — William Shakespeare (probably generated by MiniGPT at some point)
