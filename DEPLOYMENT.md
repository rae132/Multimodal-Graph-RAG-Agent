# 🌟 多模态 Graph-RAG 智能体 (OmAgent + RAG-Anything) 部署手册

本项目的核心目标是结合 **OmAgent** 的多模态工作流编排能力与 **RAG-Anything (基于 LightRAG)** 的双引擎检索能力，构建一个能够理解图片、解析复杂 PDF 文档，并基于“知识图谱 + 向量语义”进行高维度逻辑问答的终极智能助手。

经过底层的深度重构，该系统已完美适配**通义千问 (Qwen)** 等兼容 OpenAI 协议的国内大模型，彻底拔除了底层框架对单一平台的硬编码束缚。

---

## 1. 核心架构与重构亮点

1. **Graph + Vector 双引擎检索 (`Mix` 模式)**：不仅使用 FAISS 进行传统的 1024 维语义切片（Chunk）检索，更通过 NetworkX 构建并检索实体关系网（Entity-Relation Graph）。最终融合两种上下文，使智能体兼具“细节检索”与“逻辑推理”能力。
2. **FAISS 纯本地工业级向量库**：抛弃了存在 1536 维度硬编码缺陷的默认简易库，全面接入 Facebook 开源的极速本地内存向量引擎 **FAISS**，让 Qwen 等非 1536 维度的嵌入模型可以 100% 满血运行。
3. **图谱检索与生成解耦**：将 RAG 检索到的纯净结构化上下文（Context）直接透传给外层 OmAgent 工作流，由前端主控的 Qwen 智能体进行拟人化、高质量的中文回答，省去了冗余的中间总结损耗。
4. **全异步防死锁流水线**：重写了 Worker 的执行逻辑，将模型生成升级为真异步 `agenerate`，彻底消灭了 Python `asyncio` 与同步 `httpx` 在子线程中引发的死锁悬案。
5. **Mac 兼容性光环**：系统入口自动注入 `KMP_DUPLICATE_LIB_OK=TRUE`，优雅规避了 macOS 系统下常见的 OpenMP 资源抢占崩溃问题。

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

## 3. 系统配置 (使用 .env 文件)

系统的灵魂在于配置的解耦。请在 `omagent/examples/rag_multimodal_agent/.env` 目录下创建或修改该文件：

```ini
# --- 1. 聊天生成模型配置 (Chat LLM, 负责最终回答您的提问) ---
CHAT_API_KEY=您的_API_KEY
CHAT_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
CUSTOM_MODEL_ID=qwen-plus

# --- 2. 嵌入检索模型配置 (Embedding Model, 负责把文档变成向量) ---
# (您可以将其配置为与聊天模型不同的平台或本地服务)
EMBED_API_KEY=您的_API_KEY
EMBED_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
EMBED_MODEL_ID=text-embedding-v3
EMBED_DIM=1024  # [关键] 必须准确填入新模型的真实输出维度 (Qwen 为 1024)

# --- 3. RAG 引擎与解析器配置 ---
RAG_WORKING_DIR=./rag_storage
PARSER=mineru
PARSE_METHOD=auto

# --- 4. OmAgent 运行层配置 ---
OMAGENT_MODE=lite
# 兼容映射 (请与上面的 CHAT 配置保持一致)
custom_openai_key=您的_API_KEY
custom_openai_endpoint=https://dashscope.aliyuncs.com/compatible-mode/v1
```

*进阶技巧：若需极强的识图能力，可将 `CUSTOM_MODEL_ID` 更改为 `qwen-vl-max`。*

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

* **更换嵌入模型或更改维度后报错？**
  **极其重要**：如果改变了 `.env` 中的 `EMBED_DIM` 或模型，旧的 `rag_storage` 目录中的向量将无法对齐。请必须在启动前执行 `rm -rf rag_storage/` 和 `find . -name "vdb_*.json" -delete` 彻底清空旧库。
* **初次上传解析很慢？**
  若您上传的是 PDF，系统默认的 MinerU 解析器会在初次运行时下载深度学习权重（Fetching files...），视网络情况可能需要 1-3 分钟，之后将极其迅速。
* **国内网络环境下卡死？**
  代码已内置关闭 `geocoder` 位置请求（`use_default_sys_prompt: false`），杜绝了因 `ipinfo.io` 墙内访问超时引发的假死问题。
