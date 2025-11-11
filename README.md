# Modern AI Tasks with Unsloth.ai — Colab Collection

This repository contains **five Colab notebooks** demonstrating a range of **modern AI workflows** using **[Unsloth.ai](https://unsloth.ai)**.  
Each notebook includes end-to-end code and a **YouTube walkthrough** explaining the process, dataset, and results.

---

## Project Overview

| No. | Colab Notebook | Description | YouTube Walkthrough |
|-----|----------------|--------------|---------------------|
| **1️⃣** | `colab1_full_finetune.ipynb` | **Full fine-tuning** of a small LLM (`smollm2-135M`) with `full_finetuning=True`. Covers dataset prep, templates, and training pipeline. | 🎥 [Watch Colab 1 Video](http://tiny.cc/jbtu001) |
| **2️⃣** | `colab2_lora_finetune.ipynb` | **Parameter-efficient LoRA fine-tuning** using the same dataset and base model as Colab 1. Shows how LoRA reduces memory cost while maintaining performance. | 🎥 [Watch Colab 2 Video](http://tiny.cc/ibtu001) |
| **3️⃣** | `colab3_reinforcement_learning.ipynb` | **Reinforcement Learning with Preference Data** — demonstrates reward modeling with chosen/rejected responses and reinforcement-based optimization. | 🎥 [Watch Colab 3 Video](http://tiny.cc/gbtu001) |
| **4️⃣** | `colab4_grpo_reasoning.ipynb` | **Reasoning via GRPO (Group Relative Policy Optimization)**. Trains a lightweight reasoning model using numeric rewards for step-by-step problem solving. | 🎥 [Watch Colab 4 Video](http://tiny.cc/fbtu001) |
| **5️⃣** | `colab5_continued_pretrain_hindi.ipynb` | **Continued Pre-training for Language Adaptation** — extends a multilingual model to **Hindi**, including tokenization analysis, LoRA pretraining, and text generation tests. | 🎥 [Watch Colab 5 Video](http://tiny.cc/ebtu001) |

---

## Requirements

Each notebook is built for Google Colab (GPU runtime).  
They automatically install the latest versions of:
- `unsloth`
- `transformers`
- `trl`
- `peft`
- `datasets`
- `accelerate`
- `bitsandbytes`

> Tip: For gated datasets (e.g., OSCAR 23.01), log in with your [Hugging Face token](https://huggingface.co/settings/tokens).

---

## Datasets Used

| Notebook | Primary Dataset | Purpose |
|-----------|----------------|----------|
| Colab 1–2 | Custom small chat/coding dataset | Supervised fine-tuning |
| Colab 3 | Reward dataset with preferred / rejected responses | RLHF-style training |
| Colab 4 | GSM8K / ASDiv (math reasoning tasks) | GRPO reasoning and reward modeling |
| Colab 5 | CC100 / OSCAR / Wikipedia (Hindi text) | Continued pre-training on Hindi |

---

## Learning Outcomes

After completing these notebooks, you’ll understand how to:
- Fine-tune small LLMs efficiently using Unsloth.ai  
- Use **LoRA** for lightweight adaptation  
- Apply **Reinforcement Learning** and **GRPO** for alignment and reasoning  
- Perform **continued pre-training** to teach models new languages or domains  
- Export Unsloth models to **Ollama / GGUF** for local inference  

---

## Author Notes

This project was developed as part of an academic assignment to demonstrate practical use-cases of **Unsloth.ai** for model fine-tuning, RL, and adaptation.

> Each notebook runs independently in Colab.  
> For detailed explanation, see the linked YouTube videos corresponding to each notebook.

---

## License

This repository is for educational and research purposes only.  
Feel free to fork and adapt for your own experiments with attribution.

