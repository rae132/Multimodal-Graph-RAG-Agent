from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
from pydantic import BaseModel, Field
import os
import asyncio
import logging as std_logging
import numpy as np
import aiohttp
import json

from raganything import RAGAnything, RAGAnythingConfig
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

from ....utils.registry import registry
from ...base import BaseTool

logger = std_logging.getLogger(__name__)

class RAGAnythingArgs(BaseModel):
    query: Optional[str] = Field(None, description="The query string to search in the RAG knowledge base.")
    mode: str = Field("hybrid", description="Query mode: 'naive', 'local', 'global', 'hybrid', 'mix'.")
    image_paths: Optional[List[str]] = Field(None, description="Optional list of image paths for multimodal query enhancement.")
    action: str = Field("query", description="Action to perform: 'query', 'upload', 'list_docs', or 'delete'.")
    file_path: Optional[str] = Field(None, description="File path for 'upload' action.")
    doc_id: Optional[str] = Field(None, description="Document ID for 'delete' action.")

@registry.register_tool()
class RAGAnythingTool(BaseTool):
    args_schema: Type[BaseModel] = RAGAnythingArgs
    description: str = "A tool for multimodal RAG and knowledge graph search."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._working_dir = os.path.abspath(os.environ.get("RAG_WORKING_DIR", "./rag_storage"))
        
    def _get_rag_instance(self):
        # We must NOT cache `self._rag_instance` across different `_run` calls because OmAgent 
        # executes Workers in different threads/async loops. LightRAG internally uses 
        # `asyncio.Lock()` which binds to the specific event loop active during its __init__.
        # If we cache it, a new thread's loop will try to acquire the old loop's lock and crash.
        std_logging.info(f"CRITICAL-INIT: Creating new RAG instance for current event loop at {self._working_dir}")
        
        # --- 强制重置 LightRAG 的全局锁缓存 ---
        # LightRAG 1.4.10 会把第一次运行时的 asyncio.Lock 缓存在其全局变量中 (shared_storage.py)
        # 这会导致跨线程 (多事件循环) 运行时触发 bound to a different event loop 错误。
        try:
            import lightrag.kg.shared_storage as ss
            ss._initialized = False  # 破解：强行标记为未初始化
            ss._storage_keyed_lock = None # 清空之前的 KeyedLock 实例
            ss.initialize_share_data(1)  # 强制在当前线程的 event_loop 下重新创建所有锁
        except ImportError:
            pass
        
        # --- 多模型路由配置 ---
        chat_model = os.environ.get("CHAT_MODEL_ID", os.environ.get("CUSTOM_MODEL_ID", "openai/gpt-4o"))
        chat_api_key = os.environ.get("CHAT_API_KEY", os.environ.get("OPENAI_API_KEY"))
        chat_base_url = os.environ.get("CHAT_API_BASE", os.environ.get("OPENAI_API_BASE"))
        
        extract_model = os.environ.get("EXTRACT_MODEL_ID", chat_model)
        vision_model = os.environ.get("VISION_MODEL_ID", chat_model)
        
        embed_model = os.environ.get("EMBED_MODEL_ID", "openai/text-embedding-3-large")
        embed_api_key = os.environ.get("EMBED_API_KEY", chat_api_key)
        embed_base_url = os.environ.get("EMBED_API_BASE", chat_base_url)
        
        from litellm import acompletion, aembedding
        
        # --- 自动探测 Embedding 维度 (避免人工配置错误) ---
        embed_dim = int(os.environ.get("EMBED_DIM", "0"))
        if embed_dim <= 0:
            std_logging.info(f"Auto-detecting embedding dimension for model '{embed_model}'...")
            try:
                import requests
                # Use raw request for probing to bypass Litellm's default encoding_format insertion 
                # which fails on some OpenAI-compatible endpoints like Aliyun DashScope.
                probe_url = f"{embed_base_url.rstrip('/')}/embeddings"
                headers = {
                    "Authorization": f"Bearer {embed_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": embed_model.replace("openai/", "") if embed_model.startswith("openai/") else embed_model,
                    "input": ["probe"]
                }
                probe_resp = requests.post(probe_url, headers=headers, json=payload, timeout=10)
                if probe_resp.status_code == 200:
                    probe_data = probe_resp.json()
                    if "data" in probe_data and len(probe_data["data"]) > 0:
                        embed_dim = len(probe_data["data"][0]["embedding"])
                        std_logging.info(f"Successfully auto-detected embedding dimension: {embed_dim}")
                    else:
                        raise ValueError(f"Invalid response format: {probe_data}")
                else:
                    raise ValueError(f"HTTP {probe_resp.status_code}: {probe_resp.text}")
            except Exception as e:
                std_logging.warning(f"Failed to auto-detect embedding dimension: {e}. Falling back to default 1024.")
                embed_dim = 1024
        else:
            std_logging.info(f"Using explicitly configured embedding dimension: {embed_dim}")

        async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            kwargs.pop("history", None)
            
            # --- 清理 LightRAG 内部专用的 kwargs，防止被 Litellm 误当作 API 参数序列化报错 ---
            kwargs.pop("hashing_kv", None)
            kwargs.pop("openai_client_configs", None)
            keyword_extraction = kwargs.pop("keyword_extraction", False)
            if keyword_extraction:
                # LightRAG keyword extraction needs a specific JSON response format if supported by the model
                from lightrag.llm.openai import GPTKeywordExtractionFormat
                kwargs["response_format"] = GPTKeywordExtractionFormat

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.extend(history_messages)
            messages.append({"role": "user", "content": prompt})

            # --- 动态路由逻辑 ---
            # LightRAG 在构建知识图谱时，系统提示词中通常会包含抽取实体和关系的指令
            # 通过判断提示词内容，自动切换到专门的抽取模型 (如 DeepSeek 等更便宜或更擅长抽取的模型)
            current_model = chat_model
            if system_prompt and any(kw in system_prompt.lower() for kw in ["entity", "relationship", "graph", "extract", "抽取"]):
                current_model = extract_model

            resp = await acompletion(
                model=current_model,
                messages=messages,
                api_key=chat_api_key,
                api_base=chat_base_url,
                **kwargs
            )
            return resp.choices[0].message.content

        # Vision function for Multimodal Queries
        async def vision_func(prompt, system_prompt=None, image_data=None, messages=None, **kwargs):
            if messages is None:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                user_content = [{"type": "text", "text": prompt}]
                if image_data:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    })
                messages.append({"role": "user", "content": user_content})
            
            resp = await acompletion(
                model=vision_model,
                messages=messages,
                api_key=chat_api_key,
                api_base=chat_base_url,
                temperature=0.0,
                **kwargs
            )
            return resp.choices[0].message.content

        # Litellm 统一的 EMBEDDING FUNCTION
        async def embed_func(texts: List[str]) -> np.ndarray:
            resp = await aembedding(
                model=embed_model,
                input=texts,
                api_key=embed_api_key,
                api_base=embed_base_url,
                encoding_format="float" # 强制使用 float，解决阿里云等接口不支持 litellm 默认的 base64 编码问题
            )
            embeddings = [item['embedding'] for item in resp.data]
            return np.array(embeddings, dtype=np.float32)
        
        embedding_obj = EmbeddingFunc(
            embedding_dim=embed_dim,
            max_token_size=8192,
            func=embed_func
        )

        lrag = LightRAG(
            working_dir=self._working_dir,
            llm_model_func=llm_func,
            embedding_func=embedding_obj,
            llm_model_name=chat_model,
            vector_storage="FaissVectorDBStorage"
        )

        config = RAGAnythingConfig(
            working_dir=self._working_dir,
            parser=os.environ.get("PARSER", "mineru")
        )

        rag_instance = RAGAnything(config=config, lightrag=lrag, vision_model_func=vision_func)
        std_logging.info(f"SUCCESS: System armed with FAISS natively supporting {embed_dim} dimensions and Litellm Multi-Model Routing.")
            
        return rag_instance

    async def _ensure_init(self, rag):
        """Ensure LightRAG storages are initialized (locks, etc.) before any operation."""
        if rag.lightrag is not None:
            if hasattr(rag.lightrag, "initialize_storages"):
                await rag.lightrag.initialize_storages()
        else:
            await rag._ensure_lightrag_initialized()
        if hasattr(rag, "initialize_storages"):
            await rag.initialize_storages()
        return rag

    def _run(self, query: str = None, mode: str = "hybrid", image_paths: List[str] = None, action: str = "query", file_path: str = None, doc_id: str = None) -> Union[str, Dict[str, Any], List[Any]]:
        new_loop_created = False
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop_created = True
            
        try:
            return loop.run_until_complete(self._arun(query, mode, image_paths, action, file_path, doc_id))
        finally:
            if new_loop_created:
                try:
                    asyncio.set_event_loop(None)
                    loop.close()
                except Exception:
                    pass

    async def _arun(self, query: str = None, mode: str = "hybrid", image_paths: List[str] = None, action: str = "query", file_path: str = None, doc_id: str = None) -> Union[str, Dict[str, Any], List[Any]]:
        rag = self._get_rag_instance()
        await self._ensure_init(rag)
        
        if action == "upload":
            if not file_path or not os.path.exists(file_path):
                return f"Error: File path '{file_path}' does not exist."
            try:
                await rag.process_document_complete(file_path)
                return f"Successfully uploaded and processed: {file_path}"
            except Exception as e:
                logger.error(f"RAG upload error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return f"Error during processing: {str(e)}"
            
        elif action == "query":
            try:
                if image_paths and len(image_paths) > 0:
                    multimodal_content = [{"type": "image", "img_path": img} for img in image_paths if os.path.exists(img)]
                    if multimodal_content:
                        return await rag.aquery_with_multimodal(query, multimodal_content, mode=mode, only_need_context=True)
                
                return await rag.aquery(query, mode=mode, only_need_context=True)
            except Exception as e:
                logger.error(f"RAG query error: {e}")
                return f"Error during query: {str(e)}"
        
        elif action == "list_docs":
            try:
                docs = []
                # Primary source: text_chunks storage (most reliable for what's actually in DB)
                if hasattr(rag.lightrag, "text_chunks") and hasattr(rag.lightrag.text_chunks, "_data"):
                    doc_map = {}
                    import time
                    for cid, chunk in rag.lightrag.text_chunks._data.items():
                        did = chunk.get("full_doc_id")
                        if did and did not in doc_map:
                            doc_map[did] = {
                                "doc_id": did,
                                "file_name": chunk.get("file_path", "Unknown"),
                                "status": "processed",
                                "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(chunk.get("create_time", 0))) if chunk.get("create_time") else "N/A"
                            }
                    docs = list(doc_map.values())
                
                # Fallback to doc_status if chunks are empty
                if not docs:
                    status_storage = rag.lightrag.doc_status
                    data = {}
                    if hasattr(status_storage, "_data") and status_storage._data:
                        data = status_storage._data
                    
                    if data:
                        for did, info in data.items():
                            docs.append({
                                "doc_id": did,
                                "file_name": os.path.basename(info.get("file_path", info.get("content_summary", "Unknown"))),
                                "status": info.get("status", "unknown"),
                                "created_at": info.get("created_at", "N/A")
                            })
                
                logger.info(f"RAG list_docs found {len(docs)} documents.")
                return docs
            except Exception as e:
                logger.error(f"Error listing docs: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return []

        elif action == "delete":
            if not doc_id:
                return "Error: Document ID is required for delete action."
            try:
                # 1. Aggressive Doc ID cleanup: Ensure LightRAG knows about this doc before deleting
                # Often doc_status is empty but text_chunks still contains data
                chunk_ids = []
                file_path = "Unknown"
                if hasattr(rag.lightrag, "text_chunks") and hasattr(rag.lightrag.text_chunks, "_data"):
                    for cid, chunk in rag.lightrag.text_chunks._data.items():
                        if chunk.get("full_doc_id") == doc_id:
                            chunk_ids.append(cid)
                            file_path = chunk.get("file_path", file_path)
                
                # Inject status if missing to satisfy LightRAG's internal check
                status_entry = await rag.lightrag.doc_status.get_by_id(doc_id)
                if not status_entry and chunk_ids:
                    logger.info(f"RAG: Injecting missing doc_status for {doc_id} to enable deletion.")
                    await rag.lightrag.doc_status.upsert({
                        doc_id: {
                            "status": "processed",
                            "file_path": file_path,
                            "chunks_list": chunk_ids
                        }
                    })

                # 2. Perform actual deletion
                if hasattr(rag.lightrag, "adelete_by_doc_id"):
                    await rag.lightrag.adelete_by_doc_id(doc_id)
                    
                    # 3. CRITICAL: Manually trigger persistence for all storages
                    persistence_tasks = []
                    storage_attrs = [
                        "text_chunks", "full_entities", "full_relations", 
                        "full_docs", "doc_status", "llm_response_cache"
                    ]
                    for attr in storage_attrs:
                        storage = getattr(rag.lightrag, attr, None)
                        if storage and hasattr(storage, "index_done_callback"):
                            persistence_tasks.append(storage.index_done_callback())
                    
                    for vdb_attr in ["chunks_vdb", "entities_vdb", "relationships_vdb"]:
                        vdb = getattr(rag.lightrag, vdb_attr, None)
                        if vdb and hasattr(vdb, "index_done_callback"):
                            persistence_tasks.append(vdb.index_done_callback())
                            
                    if persistence_tasks:
                        await asyncio.gather(*persistence_tasks)
                        
                    return f"Successfully deleted document: {doc_id} and persisted changes to disk."
                return "Error: Delete functionality not available in this LightRAG version."
            except Exception as e:
                logger.error(f"Error deleting doc {doc_id}: {e}")
                return f"Error during deletion: {str(e)}"
            
        return "Error: Invalid action."
