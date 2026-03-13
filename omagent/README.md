<p align="center">
  <img src="docs/images/OmAgent-banner.png" width="400"/>
</p>

<div>
    <h1 align="center">🌟 Multimodal Graph RAG Agent 🌟</h1>
</div>

<p align="center">
  <a href="https://twitter.com/intent/follow?screen_name=OmAI_lab" target="_blank">
    <img alt="X (formerly Twitter) Follow" src="https://img.shields.io/twitter/follow/OmAI_lab">
  </a>
</p>

## 📖 Introduction  
**Multimodal Graph RAG Agent** is an advanced AI system built on top of the OmAgent framework. It integrates **Knowledge Graph-based RAG (Graph RAG)** with a suite of **9 different reasoning architectures**, enabling robust, grounded, and multi-step reasoning over both text and visual inputs.

This project focuses on providing a **Unified Entry Point** for researchers and developers to experiment with state-of-the-art agentic workflows while maintaining a persistent knowledge base that includes multimodal data (images, documents, etc.).

## 🔑 Key Features  
 - **🚀 Unified CLI Interface**: A single script (`start_system_cli.py`) to manage knowledge and launch chats with any supported agent engine.
 - **🧠 9 Reasoning Engines**: Built-in support for CoT, ToT, ReAct, React_Pro, Reflexion, PoT, SC_CoT, DnC, and RAP.
 - **🕸️ Graph-Based RAG**: Utilizes `LightRAG` and `RAGAnything` to build a persistent knowledge graph from your documents, supporting hybrid and multimodal retrieval.
 - **🖼️ Multimodal Support**: Native handling of image inputs across all reasoning engines. Images are cached and accessible to vision-capable models (e.g., Qwen-VL, GPT-4o).
 - **📉 Lite Mode**: Runs entirely in local mode using `redislite` and local file storage, eliminating the need for complex server deployments.

## 🛠️ Quick Start

### 1. Installation
```bash
conda create -n omagent python=3.11
conda activate omagent
pip install -e omagent-core
pip install raganything lightrag-hku litellm redis redislite func-timeout sympy statistics json_repair geocoder
```

### 2. Configuration
Create a `.env` file in the root directory:
```bash
CHAT_MODEL_ID=openai/qwen-plus
CHAT_API_KEY=your_key
CHAT_API_BASE=https://api.openai.com/v1

EMBED_MODEL_ID=openai/text-embedding-v3
EMBED_API_KEY=your_key
EMBED_API_BASE=https://api.openai.com/v1
```

### 3. Run the System
```bash
python start_system_cli.py
```

## 📂 Project Structure
- **`start_system_cli.py`**: The main entry point.
- **`system_workers.py`**: Implementation of system-level workers (Input, RAG, Menu).
- **`system_configs/`**: Unified configuration for all workers and LLM mappings.
- **`rag_storage/`**: Where your knowledge graph and vector database live.
- **`examples/`**: Reference implementations for individual architectures.

## 📖 Documentation
- [Detailed Deployment Guide](./DEPLOYMENT.md)
- [OmAgent Core Documentation](https://om-ai-lab.github.io/OmAgentDocs/)

## 🤝 Contributing
Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## ⭐️ Citation
If you find this project useful, please cite:
```bibtex
@article{zhang2024omagent,
  title={OmAgent: A Multi-modal Agent Framework for Complex Video Understanding with Task Divide-and-Conquer},
  author={Zhang, Lu and Zhao, Tiancheng and Ying, Heting and Ma, Yibo and Lee, Kyusong},
  journal={arXiv preprint arXiv:2406.16620},
  year={2024}
}
```
