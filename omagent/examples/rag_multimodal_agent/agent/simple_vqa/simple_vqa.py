from typing import List, Optional, Any
from pydantic import Field
from omagent_core.utils.registry import registry
from omagent_core.engine.worker.base import BaseWorker
from omagent_core.tool_system.manager import ToolManager
from omagent_core.utils.logger import logging

@registry.register_worker()
class RAGMultimodalWorker(BaseWorker):
    llm: Any  # Allow dict or BaseLLM

    def _get_llm_instance(self):
        if isinstance(self.llm, dict):
            llm_config = self.llm
            llm_name = llm_config.get("name")
            if not llm_name:
                raise ValueError("LLM config must have a 'name' field")
            llm_cls = registry.get_llm(llm_name)
            self.llm = llm_cls(**llm_config)
        return self.llm

    def _run(self, query: str, image_paths: List[str] = None, **kwargs):
        print(f"\n[DEBUG] RAGMultimodalWorker started with query: {query}")
        
        # 1. Use RAG tool to get context directly via sync call
        try:
            from omagent_core.tool_system.tools.rag_anything.rag_anything import RAGAnythingTool
            rag_tool = RAGAnythingTool()
            print("[DEBUG] Calling RAGAnythingTool._run...")
            rag_res = rag_tool._run(
                query=query, 
                image_paths=image_paths, 
                mode="mix",
                action="query"
            )
            print("[DEBUG] RAGAnythingTool._run completed.")
        except Exception as e:
            print(f"[DEBUG] Error executing RAGAnythingTool: {e}")
            import traceback
            traceback.print_exc()
            rag_res = f"Error retrieving information: {str(e)}"
        
        rag_res_str = str(rag_res)
        print(f"[DEBUG] RAG Retrieval complete. Result length: {len(rag_res_str)}")
        
        prompt = f"""你是一个专业且严谨的人工智能助手。请你仔细阅读下方提供的检索信息（包括文本图谱和可能包含的图片描述），并基于这些信息来回答用户的问题。

【背景检索信息】
{rag_res_str}

【用户问题】
{query}

【回答要求】
1. 请务必使用 **中文 (Chinese)** 进行回答。
2. 你的回答必须清晰、准确，且严格基于上述提供的检索信息。
3. 如果检索信息中没有提供足够的线索来回答问题，请如实回答：“抱歉，基于当前的知识库检索不到相关信息。”，切勿自行编造。
"""
        from omagent_core.models.llms.schemas import Message
        from PIL import Image
        import os
        
        llm = self._get_llm_instance()
        
        content_list = [prompt]
        if image_paths and len(image_paths) > 0:
            for img_path in image_paths:
                if img_path and os.path.exists(img_path):
                    content_list.append(Image.open(img_path))
                    
        records = [Message.user(content_list)]
        
        print("[DEBUG] Sending request to LLM (Qwen)...")
        try:
            # Use SYNC generation (which is safe now because RAGAnythingTool closed its loop)
            response = llm.generate(records)
            print("[DEBUG] LLM generation complete.")
        except Exception as e:
            print(f"[DEBUG] LLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            response = f"LLM generation failed: {str(e)}"
            
        if hasattr(response, 'choices') and len(response.choices) > 0:
            content = response.choices[0].message.content
        elif isinstance(response, dict) and 'choices' in response and len(response['choices']) > 0:
            content = response['choices'][0]['message']['content']
        elif isinstance(response, list) and len(response) > 0 and 'text' in response[0]:
            content = response[0]['text']
        else:
            content = str(response)

        print(f"[DEBUG] Final output to UI: {content[:100]}...")
        
        # Send the answer via the callback
        try:
            cb = getattr(self, "callback", None)
            if cb:
                print(f"[DEBUG] Sending answer to UI via callback for instance {self.workflow_instance_id}...")
                cb.send_answer(self.workflow_instance_id, msg=content)
            else:
                print("[DEBUG] No callback component found. Skipping send_answer.")
        except ValueError as ve:
            print(f"[DEBUG] Callback component not registered. Skipping send_answer.")
        except Exception as cb_err:
            print(f"[DEBUG] Failed to send answer to UI via callback: {cb_err}")
                
        return {"last_output": content}
