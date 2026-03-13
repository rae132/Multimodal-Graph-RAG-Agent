# 🌟 多模态 Graph-RAG 智能体部署手册

本项目的核心目标是结合 **OmAgent** 的多模态工作流编排能力与 **RAG-Anything** 的双引擎检索能力，构建一个能够理解图片、解析复杂 PDF 文档，并基于“知识图谱 + 向量语义”进行高维度逻辑问答的智能助手。

---

## 1. 核心亮点

1. **多模型自由路由**：底层通过 `litellm` 代理。您可以为**聊天问答、图谱抽取、多模态视觉、向量嵌入、重排序**独立指派模型。
2. **嵌入维度自适应 (Auto-Detect)**：系统自动探测 Embedding 模型维度，无需手动配置。
3. **混合检索 (`Mix` 模式)**：融合 FAISS 语义检索与 NetworkX 实体关系检索。
4. **全异步防死锁**：彻底解决 `asyncio` 在多线程环境下的冲突。
5. **中文回复优化**：内置强制中文引导，确保所有推理引擎（CoT/ReAct等）稳定输出中文。

---

## 2. 环境准备

### 2.1 基础环境
强制要求 **Python 3.11**。

```bash
# 创建并激活环境
conda create -n multi-agent python=3.11 -y
conda activate multi-agent

# 安装依赖
pip install -r requirements.txt
```

### 2.2 源码安装 (可编辑模式)
必须安装以下组件以确保自定义逻辑生效：

```bash
# 1. 安装 omagent-core
cd omagent/omagent-core
pip install -e .

# 2. 安装 rag-anything
cd ../../rag-anything
pip install -e .
```

---

## 3. 系统配置

### 3.1 环境变量配置
进入 `omagent` 目录并创建 `.env` 文件：

```bash
cd omagent
cp .env.example .env
```

### 3.2 关键变量说明

```ini
# 主模型配置
CHAT_API_KEY=您的_API_KEY
CHAT_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL_ID=openai/qwen-plus

# 视觉模型 (用于图片理解)
VISION_MODEL_ID=openai/qwen-vl-plus

# 嵌入模型
EMBED_API_KEY=您的_API_KEY
EMBED_MODEL_ID=openai/text-embedding-v3
EMBED_DIM=0  # 0 为自动探测

# 重排序 (可选)
ENABLE_RERANK=true
RERANK_PROVIDER=aliyun
RERANK_MODEL_ID=gte-rerank-v2
```

---

## 4. 运行系统

### 4.1 Web 界面 (推荐)
```bash
cd omagent
python run_webpage.py
```
访问 `http://127.0.0.1:7860` 即可开始。

### 4.2 命令行界面
```bash
cd omagent
python start_system_cli.py
```

---

## 5. 常见问题 (FAQ)

* **更换嵌入模型后报错？**
  更换模型或维度后，必须删除 `omagent/rag_storage` 目录以清空旧索引，否则会报维度不匹配错误。
* **解析 PDF 变慢？**
  首次运行 MinerU 解析器会下载模型权重，请耐心等待 1-3 分钟。
* **回答仍是英文？**
  请确保 `.env` 中的 `CHAT_MODEL_ID` 配置正确，并检查 `omagent-core` 中的提示词模板已包含中文引导（本项目已默认包含）。
