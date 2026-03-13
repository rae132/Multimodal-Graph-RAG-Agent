# 🤖 Multimodal Graph-RAG Agent (基于 OmAgent + RAG-Anything)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue.svg" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Framework-OmAgent-orange.svg" alt="OmAgent">
  <img src="https://img.shields.io/badge/Engine-LightRAG-green.svg" alt="LightRAG">
  <img src="https://img.shields.io/badge/VectorDB-FAISS-red.svg" alt="FAISS">
  <img src="https://img.shields.io/badge/Router-Litellm-blueviolet.svg" alt="Litellm">
</p>

本项目是一个融合了**多模态感知 (Vision)**、**知识图谱 (Graph RAG)** 与 **工业级向量检索 (Vector RAG)** 的下一代智能体系统。它能够“看懂”复杂的 PDF 文档（含图表），自动构建底层语义图谱，并支持用户通过“文字+图片”的混合方式进行高维度的专业逻辑问答。

---

## 🌟 核心特性

- **🚀 多模型智能路由 (Powered by Litellm)**：
  - 彻底解除平台绑定，**聊天、图谱抽取、视觉解析、向量嵌入、重排序**五大任务均可独立配置不同厂商的模型（OpenAI/Claude/DeepSeek/Qwen/本地模型等），实现成本与性能的极致平衡。
- **🔍 双引擎混合检索 (`Mix` Mode)**：
  - **Graph RAG**: 基于实体关系网的深层逻辑推理。
  - **Vector RAG**: 基于 FAISS 的高精度语义匹配，且**支持启动时自动探测 Embedding 维度**。
- **📸 真正多模态交互**：
  - 支持上传截图提问，自动识别图中内容并结合图谱库回答。
- **📊 交互式知识图谱可视化**：
  - 内置基于 `vis-network` 的动态渲染界面，实时展示知识间的拓扑关联。
- **⚡ 极致工程优化**：
  - **全异步流水线**: 彻底解决 Python 异步死锁导致的 UI 挂起。
  - **多样化引擎**: 内置 CoT, ToT, ReAct, Reflexion 等多种推理策略。

---

## 🛠️ 技术架构

| 模块 | 技术选型 |
| :--- | :--- |
| **智能体框架** | OmAgent (Workflows & Workers) |
| **RAG 引擎** | RAG-Anything (Powered by LightRAG) |
| **模型网关** | Litellm (Any Model, Standard API) |
| **向量库** | FAISS (purely local & auto-dimension) |
| **多模态解析** | MinerU (Magic-PDF) |
| **Web 界面** | Gradio (Custom Multitab UI) |

---

## 🚀 快速开始

### 1. 环境安装
```bash
conda create -n multi-agent python=3.11 -y
conda activate multi-agent
pip install -r requirements.txt

# 以编辑模式安装核心组件
cd omagent/omagent-core && pip install -e .
cd ../../rag-anything && pip install -e .
```

### 2. 配置 API Key
1. 进入 `omagent/` 目录。
2. 复制模板：`cp .env.example .env`。
3. 修改 `.env` 填入您的 API Key 和模型配置。

### 3. 启动项目
- **Web 界面（推荐）**:
  ```bash
  cd omagent
  python run_webpage.py
  ```
- **命令行界面**:
  ```bash
  cd omagent
  python start_system_cli.py
  ```

---

## 📸 界面预览

| 💬 智能问答 | 📚 知识库管理 | 🕸️ 知识图谱展示 |
| :---: | :---: | :---: |
| ![](doc/Images/效果图1.png) | ![](doc/Images/效果图2.png) | ![](doc/Images/知识图谱.png) |

---

## 🧗‍♂️ 项目难点与攻克记录
在整合过程中，我们解决了 **Python 异步锁跨线程死锁**、**动态 Embedding 维度探测**、**API 兼容性清洗**、**中文回复强制引导**等多个底层架构冲突。

---

## 🤝 贡献与感谢

本项目基于以下优秀的开源项目重构而成：
- [OmAgent](https://github.com/omagent-io/omagent): 灵活的多模态智能体框架。
- [RAG-Anything](https://github.com/Bob-Zheng/rag-anything): 全能多模态 RAG 工具。
- [LightRAG](https://github.com/HKU-Smart-AILab/LightRAG): 极致的图谱 RAG 算法。
- [Litellm](https://github.com/BerriAI/litellm): 统一模型调用网关。

---
<p align="center">Made with ❤️ for Multimodal AI Enthusiasts</p>
