# LLMxCPG Inference

Run vulnerability detection using LLMxCPG models on several benchmark datasets.

---

## ✅ Installation

Create and activate the conda environment:

```bash
conda create --name llmxcpg \
    python=3.11 \
    pytorch-cuda=12.4 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y

conda activate llmxcpg
```

Then install Python dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Running inference

Run inference on any supported dataset:

```bash
python3 detect_inference.py <dataset> --base-model QCRI/LLMxCPG-D
```

Datasets: `formai`, `primevul`, `reposvul`, `pkco`, `sven`.

Example:

```bash
python3 detect_inference.py formai --base-model /workspace/QCRI__LLMxCPG-D
```
