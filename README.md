# Whisper Fine-Tuning for Code-Switching Speech Recognition

This project develops a robust ASR system for code-switched speech using the **SEAME dataset**, which contains Mandarin-English conversational recordings. We adapt a pre-trained **Whisper-small** model using three strategies: **LoRA fine-tuning**, **zero-shot**, and **few-shot in-context learning**.

We evaluate performance with **WER** (English), **CER** (Mandarin), and **MER** (overall), while also tracking computational efficiency (trainable parameters, GPU memory, training time, checkpoint size).

**Expected outcome:** LoRA achieves higher accuracy near code-switch points, while few-shot in-context learning offers a flexible, training-free alternative under limited resources.
