## Whisper Fine-Tuning for Code-Switching Speech Recognition


## **Project Overview**

This project evaluates **Whisper-small** for **code-switched speech recognition**, focusing on two Mandarin-English SEAME datasets:

- [SEAME Dev SGE](https://huggingface.co/datasets/AudioLLMs/seame_dev_sge)  
- [SEAME Dev MAN](https://huggingface.co/datasets/AudioLLMs/seame_dev_man)

We investigate two approaches to improve ASR performance on code-switched speech:

1. **LoRA Fine-Tuning** – Adapting Whisper with Low-Rank Adapters.
2. **In-Context Learning (ICL)** – Zero-shot and few-shot adaptation without model updates.

We compare the approaches in terms of **accuracy (WER/CER/MER)** and **efficiency (training/inference time, memory usage)**.


## Repository Structure

```
   whisper-fine-tuning-for-code-switched-asr/
   ├── README.md
   ├── LICENSE
   ├── requirements.txt
   ├── data/
   │   ├── raw/
   │   └── processed/
   ├── scripts/
   ├── notebooks/
   │   ├── 01_data_exploration.ipynb
   │   ├── 02_pretrained_whisper_evaluation.ipynb
   │   ├── 03_lora_fine_tuning.ipynb
   │   ├── 04_in_context_learning.ipynb
   │   └── 05_comparison_analysis.ipynb
   ├── src/
   │   ├── data/
   │   ├── models/
   │   ├── analysis/
   │   └── config.py
   ├── tests/
   ├── results/

```

> All main experiments and analysis are implemented in the `notebooks/` folder.


## **Setup Instructions**

1. Clone the repository:
```bash
git clone https://github.com/tingwang12/whisper-fine-tuning-for-code-switched-asr.git
cd whisper-fine-tuning-for-code-switched-asr
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the datasets using Hugging Face:
```bash
from datasets import load_dataset

dataset_sge = load_dataset("AudioLLMs/seame_dev_sge")
dataset_man = load_dataset("AudioLLMs/seame_dev_man")
```

## **Workflow (Notebook-Based)**

**1. Data Exploration** (01_data_exploration.ipynb)

- Inspect audio durations, transcription lengths, and code-switch frequency.
- Preprocess audio to Whisper-compatible format (16 kHz, mono).


**2. Pretrained Whisper Evaluation** (02_pretrained_whisper_evaluation.ipynb)

- Run pretrained Whisper-small on SEAME datasets.
- Compute WER (Word Error Rate), CER (Character Error Rate), and MER (Mixed Error Rate).


**3. LoRA Fine-Tuning** (03_lora_fine_tuning.ipynb)

- Adapt Whisper-small using Low-Rank Adapters.
- Save adapter checkpoints instead of the full model.
- Evaluate WER/CER/MER after fine-tuning.


**4. In-Context Learning (ICL)** (04_in_context_learning.ipynb)

- Zero-shot: Use only instructions without examples.
- Few-shot: Provide a few code-switched audio-transcript pairs as context.
- Evaluate WER/CER/MER on test splits.


**5. Comparison Analysis** (05_comparison_analysis.ipynb)

- Compare accuracy (WER/CER) and efficiency (training/inference time, memory usage).
- Visualize results with tables and plots.

