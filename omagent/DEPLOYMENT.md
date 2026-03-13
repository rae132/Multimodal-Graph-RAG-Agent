# Deployment Guide

This guide provides instructions for deploying and configuring the Multimodal-Graph-RAG-Agent system.

## 📋 Prerequisites

- **Python**: 3.10 or higher.
- **Conda**: Recommended for environment management.
- **Redis**: Required for state management (STM). The system defaults to `redislite` (Lite mode), so no manual installation is strictly necessary for local testing.
- **API Keys**: Access to OpenAI-compatible LLM and Embedding models (e.g., GPT-4o, Qwen-Plus, Text-Embedding-3-Large).

## 🛠️ Installation

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd Multimodal-Graph-RAG-Agent/omagent
   ```

2. **Setup Conda Environment**:
   ```bash
   conda create -n omagent python=3.11
   conda activate omagent
   pip install -e omagent-core
   # Install additional dependencies for RAG and specialized workers
   pip install raganything lightrag-hku litellm redis redislite func-timeout sympy statistics json_repair geocoder
   ```

## ⚙️ Configuration

The system is controlled via a root-level `.env` file. Create a `.env` file in the `omagent` directory based on the following template:

```bash
# ==========================================
#  LLM & Embedding Configuration
# ==========================================
# Use provider/model format (e.g., openai/gpt-4o, openai/qwen-plus)
# The system automatically handles prefix stripping for native OpenAI SDK calls.
CHAT_MODEL_ID=openai/qwen-plus
CHAT_API_KEY=your_chat_api_key
CHAT_API_BASE=https://api.openai.com/v1 # Or your proxy/DashScope endpoint

EMBED_MODEL_ID=openai/text-embedding-v3
EMBED_API_KEY=your_embed_api_key
EMBED_API_BASE=https://api.openai.com/v1

# ==========================================
#  RAG Configuration
# ==========================================
RAG_WORKING_DIR=./rag_storage
PARSER=mineru # Options: mineru, marker, etc.
OMAGENT_MODE=lite # Ensures system runs without heavy middleware
```

## 🚀 Running the System

### 1. Unified Entry Point
The main entry point is `start_system_cli.py`. It provides a menu-driven interface to manage knowledge and start chats with different reasoning engines.

```bash
python start_system_cli.py
```

### 2. Knowledge Management
To add documents or images to the RAG knowledge base:
1. Run `python start_system_cli.py`.
2. Select `2. Manage Knowledge`.
3. Provide the **absolute path** to your file (e.g., `/home/user/docs/manual.pdf` or `/home/user/images/diagram.png`).
4. The system will parse, extract entities/relations, and build a searchable knowledge graph in `rag_storage`.

### 3. Chatting with Agents
To query the system using a specific reasoning architecture:
1. Run `python start_system_cli.py`.
2. Select `1. Chat with Agent`.
3. Choose a reasoning engine (1-9):
   - **CoT**: Chain of Thought
   - **ToT**: Tree of Thoughts (Search-based)
   - **ReAct**: Thought-Action-Observation loop
   - **React_Pro**: Advanced ReAct with separate steps
   - **Reflexion**: Self-correcting reasoning
   - **PoT**: Program of Thoughts (Python code execution)
   - **SC_CoT**: Self-Consistency CoT (Voting)
4. Enable RAG when prompted to ground the answer in your uploaded knowledge.
5. Enter your question. For multimodal queries, include the image path: `Describe this image: /path/to/image.jpg`.

## 📂 System Structure

- `start_system_cli.py`: Main workflow orchestrator.
- `system_workers.py`: Shared workers for input, RAG retrieval, and knowledge management.
- `system_configs/`: Unified YAML configurations for all workers and LLMs.
- `rag_storage/`: Persistent storage for the knowledge graph and vector indices.
- `omagent-core/`: Core library containing engine implementations.
