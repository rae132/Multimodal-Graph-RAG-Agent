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
        self._rag_instance = None
        self._working_dir = os.path.abspath(os.environ.get("RAG_WORKING_DIR", "./rag_storage"))
        
    def _get_rag_instance(self):
        if self._rag_instance is None:
            std_logging.info(f"CRITICAL-INIT: Using FAISS Professional DB at {self._working_dir}")
            
            chat_api_key = os.environ.get("CHAT_API_KEY", os.environ.get("OPENAI_API_KEY"))
            chat_base_url = os.environ.get("CHAT_API_BASE", os.environ.get("OPENAI_API_BASE"))
            llm_model = os.environ.get("CUSTOM_MODEL_ID", "qwen-plus")
            
            embed_api_key = os.environ.get("EMBED_API_KEY", chat_api_key)
            embed_base_url = os.environ.get("EMBED_API_BASE", chat_base_url)
            embed_model = os.environ.get("EMBED_MODEL_ID", "text-embedding-v3")
            
            # Use raw unpadded dimension
            embed_dim = int(os.environ.get("EMBED_DIM", "1024"))
            
            async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                kwargs.pop("history", None)
                return await openai_complete_if_cache(
                    llm_model, prompt, system_prompt=system_prompt, history_messages=history_messages,
                    base_url=chat_base_url, api_key=chat_api_key, **kwargs
                )

            # Vision function for Multimodal Queries
            async def vision_func(prompt, system_prompt=None, image_data=None, messages=None, **kwargs):
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=chat_api_key, base_url=chat_base_url)
                
                if messages is not None:
                    api_messages = messages
                else:
                    api_messages = []
                    if system_prompt:
                        api_messages.append({"role": "system", "content": system_prompt})
                    
                    user_content = [{"type": "text", "text": prompt}]
                    if image_data:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                        })
                    api_messages.append({"role": "user", "content": user_content})
                
                vision_model = llm_model if "vl" in llm_model.lower() else "qwen-vl-max"
                
                resp = await client.chat.completions.create(
                    model=vision_model,
                    messages=api_messages,
                    temperature=0.0
                )
                return resp.choices[0].message.content

            # RAW HTTP EMBEDDING FUNCTION
            async def embed_func(texts: List[str]) -> np.ndarray:
                url = f"{embed_base_url.rstrip('/')}/embeddings"
                headers = {
                    "Authorization": f"Bearer {embed_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {"model": embed_model, "input": texts}
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, json=payload) as resp:
                        if resp.status != 200:
                            err_text = await resp.text()
                            raise ValueError(f"Embedding API failed: {err_text}")
                        data = await resp.json()
                        embeddings = [item['embedding'] for item in data['data']]
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
                llm_model_name=llm_model,
                vector_storage="FaissVectorDBStorage"
            )

            config = RAGAnythingConfig(
                working_dir=self._working_dir,
                parser=os.environ.get("PARSER", "mineru")
            )

            self._rag_instance = RAGAnything(config=config, lightrag=lrag, vision_model_func=vision_func)
            std_logging.info(f"SUCCESS: System armed with FAISS natively supporting {embed_dim} dimensions and Vision capabilities.")
                
        return self._rag_instance

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
