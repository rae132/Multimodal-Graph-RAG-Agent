from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry
from omagent_core.utils.general import read_image
from omagent_core.utils.logger import logging
import os
from typing import Any

@registry.register_worker()
class InputInterface(BaseWorker):
    def _run(self, *args, **kwargs):
        pass

@registry.register_worker()
class SystemMenuWorker(BaseWorker):
    def _run(self, *args, **kwargs):
        menu_prompt = """Welcome to OmAgent Unified System!
Please choose an action:
1. Chat with Agent (CoT, ToT, ReAct...)
2. Manage Knowledge (Upload files to RAG)
"""
        choice = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id,
            input_prompt=menu_prompt,
        )
        choice_key = choice["messages"][-1]["content"][0]["data"].strip()
        action_type = "chat" if choice_key == "1" else "knowledge"
        return {"action_type": action_type}

@registry.register_worker()
class KnowledgeUploadWorker(BaseWorker):
    def _run(self, *args, **kwargs):
        file_path_input = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id,
            input_prompt="Please enter the absolute path of the file you want to upload to RAG:",
        )
        file_path = file_path_input["messages"][-1]["content"][0]["data"].strip()
        return {"file_path": file_path}

@registry.register_worker()
class SystemInputWorker(BaseWorker):
    def _run(self, *args, **kwargs):
        # 1. 支持的所有引擎
        agents = {
            "1": "CoT",
            "2": "ToT",
            "3": "ReAct",
            "4": "React_Pro",
            "5": "Reflexion",
            "6": "PoT",
            "7": "SC_CoT",
            # "8": "DnC",
            # "9": "RAP",
        }
        
        agent_prompt = "Choose Agent Type:\n" + "\n".join([f"{k}. {v}" for k, v in agents.items()])
        agent_choice = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id,
            input_prompt=agent_prompt,
        )
        choice_key = agent_choice["messages"][-1]["content"][0]["data"]
        agent_type = agents.get(choice_key, "CoT")

        # 2. RAG 开关
        rag_choice = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id,
            input_prompt="Enable RAG? 1. Yes, 2. No",
        )
        choice_text = rag_choice["messages"][-1]["content"][0]["data"].strip().lower()
        rag_enabled = choice_text in ["1", "yes", "y"]

        # 3. 输入
        user_input = self.input.read_input(
            workflow_instance_id=self.workflow_instance_id,
            input_prompt=f"[{agent_type} Mode] Enter question (and optional image path):",
        )

        query = ""
        image_paths = []
        content = user_input["messages"][-1]["content"]
        for item in content:
            if item["type"] == "text":
                query = item["data"]
            elif item["type"] == "image_url":
                image_path = item["data"]
                image_paths.append(image_path)
                try:
                    img = read_image(input_source=image_path)
                    self.stm(self.workflow_instance_id)["image_cache"] = {"<image_0>": img}
                except:
                    pass

        return {
            "agent_type": agent_type,
            "rag_enabled": rag_enabled,
            "query": query,
            "image_paths": image_paths
        }

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() in ("yes", "true", "t", "1", "y")

@registry.register_worker()
class RAGRetrievalWorker(BaseWorker):
    def _run(self, query: str = None, image_paths: list = None, rag_enabled: Any = True, action: str = "query", file_path: str = None, **kwargs):
        # 强制转换并记录日志
        is_rag_active = str2bool(rag_enabled)
        # 容错处理：如果 action 为 None，默认为 query
        current_action = action or "query"
        
        logging.info(f"RAGRetrievalWorker: action={current_action}, rag_enabled_raw={rag_enabled}, processed={is_rag_active}")
            
        if current_action == "query" and not is_rag_active:
            logging.info("RAG is disabled, skipping retrieval.")
            return {"result": ""}
        
        # 如果不是 upload 且 RAG 关闭，也跳过
        if current_action != "upload" and not is_rag_active:
            return {"result": ""}

        from omagent_core.tool_system.tools.rag_anything.rag_anything import RAGAnythingTool
        try:
            rag_tool = RAGAnythingTool()
            if action == "upload":
                logging.info(f"Uploading file to RAG: {file_path}")
                res = rag_tool._run(action="upload", file_path=file_path)
            else:
                res = rag_tool._run(query=query, image_paths=image_paths, mode="hybrid", action="query")
            
            return {"result": str(res)}
        except Exception as e:
            logging.error(f"RAG Error: {e}")
            return {"result": f"RAG failed: {e}"}

@registry.register_worker()
class RAGContextWorker(BaseWorker):
    def _run(self, query: str, rag_res: str = "", rag_enabled: Any = False, *args, **kwargs):
        is_rag_active = str2bool(rag_enabled)
        logging.info(f"RAGContextWorker: rag_enabled_raw={rag_enabled}, processed={is_rag_active}, has_res={bool(rag_res)}")
            
        if not is_rag_active or not rag_res or str(rag_res).strip() == "":
            return {"final_query": query}
        
        final_query = f"从知识库检索到的背景知识：\n{rag_res}\n\n用户问题：{query}\n请务必根据上述背景知识，使用简体中文回答用户问题。"
        return {"final_query": final_query}
