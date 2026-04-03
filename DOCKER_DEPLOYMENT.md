# Docker 离线部署说明

这套方案的目标是：

1. 在有网机器上完成依赖安装、`tiktoken` 缓存和 MinerU 模型预热。
2. 导出成离线可加载的 Docker 镜像。
3. 在公司内网机上通过 `docker load` 直接运行。

## 适合你当前场景的结论

你现在这台机器已经满足最关键的条件：

- 项目已经在本机 `conda` 的 `multi-agent` 环境中跑通
- 本机已经装好 Docker Desktop
- 公司上网机不可靠，不适合再临时装依赖

所以正确做法就是：

1. 只在你这台机器上构建 Docker 镜像
2. 直接导出镜像压缩包或分片
3. 用 GitHub Release 或 U 盘中转
4. 到公司内网机 `docker load`

不需要在公司上网机重新安装 Python、Conda、pip 依赖。

## 先说明一个关键前提

项目已经支持把 Python 依赖、`tiktoken` 缓存、MinerU 解析模型一起封进镜像，因此“文档解析 / 建库 / 图谱展示”可以做到离线可跑。

但“问答推理”是否能离线运行，取决于你内网机上的模型访问方式：

- 如果内网机可以访问公司内网的 OpenAI 兼容模型服务、vLLM、Ollama，这个镜像可以直接用。
- 如果内网机完全不能访问任何模型服务，当前项目的问答能力仍然无法工作，因为它本身不是本地大模型一体化仓库。

也就是说，Docker 能解决“环境依赖离线化”，不能凭空解决“推理模型来源”。

## 一、在有网机器构建镜像

先准备好 `omagent/.env`，建议至少包含：

```ini
CHAT_API_KEY=your_key
CHAT_API_BASE=http://your-internal-or-external-openai-compatible-endpoint/v1
CHAT_MODEL_ID=openai/qwen-plus

EMBED_API_KEY=your_key
EMBED_API_BASE=http://your-internal-or-external-openai-compatible-endpoint/v1
EMBED_MODEL_ID=openai/text-embedding-v3
EMBED_DIM=0

VISION_MODEL_ID=openai/qwen-vl-plus
RAG_WORKING_DIR=/app/omagent/rag_storage
TIKTOKEN_CACHE_DIR=/app/tiktoken_cache
PARSER=mineru
PARSE_METHOD=auto
OMAGENT_MODE=lite
```

然后在项目根目录执行：

```bash
chmod +x scripts/build_offline_image.sh scripts/export_docker_image.sh scripts/load_docker_image.sh
./scripts/build_offline_image.sh multimodal-graph-rag-agent offline
```

这个脚本默认会用 `docker buildx` 构建 `linux/amd64` 镜像。

这样做的原因是：

- 你当前本机是 macOS ARM
- 公司内网机大概率是 Linux x86_64
- 如果直接按本机默认架构构建，带过去可能无法运行

如果你明确知道内网机也是 ARM Linux，再把平台改成：

```bash
TARGET_PLATFORM=linux/arm64 ./scripts/build_offline_image.sh multimodal-graph-rag-agent offline
```

默认会构建镜像：

```bash
multimodal-graph-rag-agent:offline
```

这个镜像在构建阶段会做三件事：

- 安装 Python 依赖和本地源码包
- 生成 `tiktoken_cache`
- 运行一次 MinerU 预热，尽量把解析模型缓存进镜像

也就是说，构建依赖这一步完全发生在你当前这台电脑，不依赖公司上网机。

## 二、导出成可传输文件

```bash
./scripts/export_docker_image.sh multimodal-graph-rag-agent offline
```

输出目录在 `dist/`，通常会得到：

```bash
dist/multimodal-graph-rag-agent-offline.tar.gz
dist/multimodal-graph-rag-agent-offline.tar.gz.part-00
dist/multimodal-graph-rag-agent-offline.tar.gz.part-01
...
```

默认按 `1900m` 分片，主要是为了适配 GitHub Release 单文件大小限制。

## 三、是否适合上传到 GitHub

可以，但分两种方式：

### 方案 A：上传到 GitHub Release

这是最适合你当前场景的方式。

做法：

1. 在有网机器把 `dist/*.part-*` 上传到 GitHub Release 附件。
2. 在公司可上 GitHub 的机器下载这些分片。
3. 再把分片拷到内网机。
4. 在内网机执行 `docker load`。

优点：

- 不污染代码仓库
- 适合大文件
- 比直接提交到 Git 仓库稳妥
- 公司上网机只负责“下载文件”，不负责“安装环境”

### 方案 B：推送到 GitHub Container Registry

如果“目标机器本身”能访问 GitHub，那么可以直接推到 `ghcr.io`。

但你的最终内网机不能联网，所以 GHCR 只适合中转到“公司上网机”，不适合作为最终运行机的直接拉取源。

### 不建议：直接把镜像 tar 提交进代码仓库

不建议把 `tar.gz` 或分片直接 `git add` 到仓库：

- 仓库会迅速膨胀
- GitHub 普通仓库对大文件很不友好
- 后续版本迭代会非常痛苦

## 四、在内网机加载并启动

如果你拿到的是单个压缩包：

```bash
./scripts/load_docker_image.sh /path/to/multimodal-graph-rag-agent-offline.tar.gz
```

如果你拿到的是多个分片：

```bash
./scripts/load_docker_image.sh /path/to/multimodal-graph-rag-agent-offline.tar.gz.part-00 /path/to/multimodal-graph-rag-agent-offline.tar.gz.part-01
```

然后准备内网机使用的 `omagent/.env`，再启动：

```bash
docker run --name mm-graph-rag \
  -p 7860:7860 \
  --env-file /absolute/path/to/omagent/.env \
  -v /absolute/path/to/rag_storage:/app/omagent/rag_storage \
  multimodal-graph-rag-agent:offline
```

启动后访问：

```bash
http://<内网机IP>:7860
```

## 五、数据持久化建议

务必挂载这两个目录：

```bash
-v /absolute/path/to/rag_storage:/app/omagent/rag_storage
```

如果你希望把模型缓存也持久化到宿主机，可再挂：

```bash
-v /absolute/path/to/model_cache:/app/model_cache
-v /absolute/path/to/tiktoken_cache:/app/tiktoken_cache
```

通常第一版镜像已经内置了缓存，这两项不是必须。

## 六、我帮你顺手处理的容器化细节

为了让它在 Docker 和离线环境里更稳，我已经把下面这些点补上了：

- `omagent/run_webpage.py` 现在会显式读取 `omagent/.env`
- 默认设置 `TIKTOKEN_CACHE_DIR`
- Gradio 监听地址改为可通过环境变量控制，默认 `0.0.0.0:7860`
- 知识图谱页面不再依赖外网 `vis-network` CDN，改成镜像内联资源，离线也能显示

## 七、已知限制

1. 如果你的 `.env` 仍然指向公网模型接口，内网机没有外网时，问答一定失败。
2. `Dockerfile` 里已经尽量预热 MinerU，但不同版本的底层模型缓存路径可能有差异；上线前建议你在有网机器上实际跑一次“上传 PDF -> 成功解析”再导出镜像。
3. 如果你后续切换了 embedding 模型或 `EMBED_DIM`，需要清空宿主机挂载的 `rag_storage` 后重建索引。
4. 你当前本机是 Mac，目标机如果是 Linux x86_64，请继续使用默认的 `linux/amd64` 构建，不要直接构建成本机原生 ARM 镜像。
