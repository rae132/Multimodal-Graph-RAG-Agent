# 🌟 多模态 Graph-RAG 智能体 (OmAgent + RAG-Anything) 部署手册

本项目的核心目标是结合 **OmAgent** 的多模态工作流编排能力与 **RAG-Anything (基于 LightRAG)** 的双引擎检索能力，构建一个能够理解图片、解析复杂 PDF 文档，并基于“知识图谱 + 向量语义”进行高维度逻辑问答的终极智能助手。

经过底层的深度重构，该系统已接入 **Litellm**，完美适配**所有兼容 OpenAI 协议的大模型（如 OpenAI、Claude、DeepSeek、Qwen、本地 Ollama/vLLM）**，彻底拔除了底层框架对单一平台的硬编码束缚，实现完全灵活的多模型路由。

---

## 1. 核心架构与重构亮点

1. **统一模型路由与按需解耦**：底层通过 `litellm` 代理所有网络请求。您可以为**聊天问答、图谱抽取、多模态视觉、向量嵌入**等四大核心任务自由指派不同的模型（例如：用 GPT-4o 聊天，用便宜的 DeepSeek 进行大规模图谱抽取，极大降低建库成本）。
2. **嵌入维度自适应 (Auto-Detect)**：不再需要痛苦地在文档中翻找 `EMBED_DIM`。系统会在启动时发送探测请求给您的 Embedding 模型，自动计算并配置真实的输出维度给底层向量库。
3. **Graph + Vector 双引擎检索 (`Mix` 模式)**：不仅使用 FAISS 进行传统的语义切片检索，更通过 NetworkX 构建并检索实体关系网。最终融合两种上下文，使智能体兼具“细节检索”与“逻辑推理”能力。
4. **FAISS 纯本地工业级向量库**：全面接入 Facebook 开源的极速本地内存向量引擎 **FAISS**，让非 1536 维度的嵌入模型（如 Qwen 或 BGE）也能 100% 满血运行。
5. **全异步防死锁流水线**：重写了 Worker 的执行逻辑，将模型生成升级为真异步 `agenerate`，彻底消灭了 Python `asyncio` 与同步 `httpx` 在子线程中引发的死锁悬案。

---

## 2. 环境准备

### 2.1 Conda 环境配置
请确保使用 `multi-agent` 环境（强制要求 Python 3.11）：

```bash
# 创建并激活环境
conda create -n multi-agent python=3.11 -y
conda activate multi-agent

# 安装基础编译依赖与 FAISS 向量库 (也可直接运行 pip install -r requirements.txt)
conda install cmake -y 
pip install -r requirements.txt
```

### 2.2 源码安装 (可编辑模式)
为了确保重构后的核心组件生效，请按照以下顺序以可编辑模式安装：

```bash
# 1. 安装 omagent-core (包含重写后的图谱前端与异步 Worker)
cd omagent/omagent-core
pip install -e .

# 2. 安装 rag-anything (包含 FAISS 集成与多模态解析器)
cd ../../rag-anything
pip install -e .
```

---

## 3. 系统配置

本项目的配置支持两种方式：**使用 `.env` 文件**（推荐）或 **在终端直接 `export` 环境变量**。

### 方式 A：使用 `.env` 文件（推荐）
在 `omagent/examples/rag_multimodal_agent/` 目录下，系统会自动读取 `.env` 文件。您可以参考同目录下的 `.env.example` 进行配置：

```bash
cd omagent/examples/rag_multimodal_agent
cp .env.example .env
# 然后使用编辑器修改 .env 填入您的 API Key
```

### 方式 B：使用终端 `export` 命令
如果您不想创建文件，也可以在启动程序前直接在终端执行：

```bash
export CHAT_API_KEY=您的_API_KEY
export CHAT_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
export CHAT_MODEL_ID=openai/qwen-plus
# ... 其他变量同理
```

---

### 配置项详细说明：

```ini
# ==========================================
#  多模型路由配置 (Litellm 统一接口)
# ==========================================

# 1. 主聊天与推理模型 (必填)
CHAT_API_KEY=您的_API_KEY
CHAT_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
CHAT_MODEL_ID=openai/qwen-plus  # 必须带有对应的厂商前缀 (如 openai/, anthropic/ 等)

# 2. 知识图谱抽取模型 (选填)
# 如果不填，默认使用 CHAT_MODEL_ID。推荐使用专门的语言模型降低建库成本。
# EXTRACT_MODEL_ID=openai/deepseek-chat

# 3. 多模态视觉模型 (选填)
VISION_MODEL_ID=openai/qwen-vl-plus

# 4. 嵌入模型与维度 (必填)
# ⚠️ 注意：更换嵌入模型后，必须删除 rag_storage 目录重新构建索引！
EMBED_API_KEY=您的_API_KEY
EMBED_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBED_MODEL_ID=openai/text-embedding-v3
# 设置为 0 开启自动探测维度。如果探测失败，可手动填写固定值 (如 1024 或 3072)
EMBED_DIM=0

# --- RAG & OmAgent 运行配置 ---
RAG_WORKING_DIR=./rag_storage
PARSER=mineru
PARSE_METHOD=auto
OMAGENT_MODE=lite

# --- 用户对话模型 ID 设置 (映射到 gpt.yml) ---
custom_openai_key=您的_API_KEY
custom_openai_endpoint=https://dashscope.aliyuncs.com/compatible-mode/v1
```

---

## 4. 运行与操作流程

### 4.1 启动系统
进入示例目录并启动 Web 客户端（所有 Mac 冲突已被代码自动拦截）：

```bash
cd omagent/examples/rag_multimodal_agent
python run_webpage.py
```
终端无报错后，在浏览器访问：`http://127.0.0.1:7860`

### 4.2 网页端三步走指南

1. **📚 知识库录入**：
    * 切换至“知识库管理”标签。
    * 上传专业领域知识文档（支持 `.pdf`, `.txt`, `.docx`）。
    * 点击 **“🚀 开始解析并录入”**。系统将在后台完成版面分析、实体提取，并异步写入 FAISS 与图谱库，请耐心等待“处理状态”返回成功。
2. **🕸️ 图谱可视化**：
    * 切换至“知识图谱展示”标签。
    * 点击 **“🔄 刷新图谱”**，即可直观审视底层提取出的实体节点与拓扑关系。
3. **💬 多模态问答**：
    * 返回“智能问答”标签。
    * 直接输入问题并发送。体验极致的 Mix 模式检索回复。

---

## 5. 运维与排障 (FAQ)

* **网页端多次提问或上传时卡死，终端提示 `bound to a different event loop`？**
  这是因为 Python 3 的 `asyncio.Lock` 严禁跨事件循环借用。由于 OmAgent 是多线程调度工作流，我们在 `rag_anything.py` 的适配器中强行破解了 LightRAG 1.4.10 的全局单例锁缓存。请确保你在拉取代码后执行了最新的覆盖。如果偶尔复现，重启 `run_webpage.py` 即可。
* **生成图谱时报错 `Object of type JsonKVStorage is not JSON serializable`？**
  这是由于底层大模型网关 Litellm 试图序列化 LightRAG 的内部对象导致的。我们已经在适配层加入了 Kwargs 拦截清洗机制。
* **更换嵌入模型或更改维度后报错？**
  **极其重要**：一旦更改了 `EMBED_MODEL_ID` 或底层的 Embedding 维度，旧的 `rag_storage` 目录中的向量将发生严重的维度不匹配（Dimension Mismatch）致命错误。请必须在启动前执行 `rm -rf rag_storage/` 和 `find . -name "vdb_*.json" -delete` 彻底清空旧库并重新录入文档。
* **什么是“自动探测维度”失败？**
  如果在无网或者 API 接口严重限流的情况下启动系统，提示“Failed to auto-detect embedding dimension”，建议将 `.env` 文件中的 `EMBED_DIM` 从 `0` 显式更改为该模型正确的输出维度（如 Qwen 的 1024）。
* **初次上传解析很慢？**
  若您上传的是 PDF，系统默认的 MinerU 解析器会在初次运行时下载深度学习权重（Fetching files...），视网络情况可能需要 1-3 分钟，之后将极其迅速。
