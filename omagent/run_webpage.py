import os
import json
import asyncio
import logging as std_logging
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import networkx as nx

# --- 环境初始化 ---
load_dotenv()
os.environ["OMAGENT_MODE"] = "lite"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 屏蔽繁琐日志
std_logging.getLogger("asyncio").setLevel(std_logging.CRITICAL)
std_logging.getLogger("lightrag").setLevel(std_logging.ERROR)

from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.switch_task import SwitchTask
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.webpage.lite_client import WebpageClient
from omagent_core.utils.logger import logging

import system_workers

logging.init_logger("omagent", "omagent", level="INFO")
CURRENT_PATH = Path(__file__).parent.absolute()
os.environ["RAG_WORKING_DIR"] = str(CURRENT_PATH / "rag_storage")

# --- RAG 工具与 UI 辅助函数 ---
rag_tool_instance = None

def get_rag_tool():
    global rag_tool_instance
    if rag_tool_instance is None:
        from omagent_core.tool_system.tools.rag_anything.rag_anything import RAGAnythingTool
        rag_tool_instance = RAGAnythingTool()
    return rag_tool_instance

async def upload_to_rag(file):
    if file is None: return "Please select a file."
    try:
        tool = get_rag_tool()
        result = await tool._arun(action="upload", file_path=file.name)
        return result
    except Exception as e:
        return f"Error: {str(e)}"

async def list_existing_docs():
    try:
        tool = get_rag_tool()
        docs = await tool._arun(action="list_docs")
        if not docs: return []
        return [[d["file_name"], d["doc_id"], d["status"], d["created_at"]] for d in docs]
    except Exception:
        return []

def list_existing_docs_sync():
    try:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(list_existing_docs())
    except: return []

async def delete_doc(doc_id):
    if not doc_id: return "Please provide a Document ID."
    try:
        tool = get_rag_tool()
        return await tool._arun(action="delete", doc_id=doc_id)
    except Exception as e:
        return f"Delete failed: {e}"

def get_kg_visualization_html():
    kg_path = Path(os.environ["RAG_WORKING_DIR"]) / "graph_chunk_entity_relation.graphml"
    if not kg_path.exists():
        return "<div style='text-align:center;padding:50px;'><h3>🕸️ 暂无知识图谱数据</h3><p>请先在「知识库管理」中上传并解析文档。</p></div>"
    
    try:
        G = nx.read_graphml(kg_path)
        nodes, edges = [], []
        for node_id, data in G.nodes(data=True):
            label = str(node_id).replace('"', "'")
            title = str(data.get("description", data)).replace('"', "'")
            nodes.append({"id": node_id, "label": label[:15] + "..." if len(label) > 15 else label, "title": title})
            
        for u, v, data in G.edges(data=True):
            label = str(data.get("relationship", data.get("weight", ""))).replace('"', "'")
            edges.append({"from": u, "to": v, "label": label[:10] + "..." if len(label) > 10 else label})
            
        nodes_json, edges_json = json.dumps(nodes), json.dumps(edges)
        html_content = f"""
        <html>
        <head><script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script></head>
        <body><div id="mynetwork" style="width:100%;height:100vh;"></div>
        <script type="text/javascript">
            var data = {{ nodes: new vis.DataSet({nodes_json}), edges: new vis.DataSet({edges_json}) }};
            var options = {{ nodes: {{ shape: 'dot', size: 16 }}, physics: {{ stabilization: true }} }};
            new vis.Network(document.getElementById('mynetwork'), data, options);
        </script></body></html>
        """
        import html
        return f'<iframe srcdoc="{html.escape(html_content)}" style="width:100%;height:600px;border:none;"></iframe>'
    except Exception as e:
        return f"<p style='color:red;'>Error loading graph: {str(e)}</p>"

# --- 自定义 WebpageClient ---
class UnifiedWebpageClient(WebpageClient):
    def __init__(self, interactor, config_path, workers, **kwargs):
        super().__init__(interactor=interactor, config_path=config_path, workers=workers, **kwargs)
        self.agent_type = "CoT"
        self.rag_enabled = True

    def add_message(self, history, message):
        text = message.get("text", "")
        files = message.get("files", [])
        
        for x in files: history.append({"role": "user", "content": {"path": x}})
        if text: history.append({"role": "user", "content": text})
            
        import threading
        if self._workflow_instance_id is None: self._workflow_instance_id = self.workflow_instance_id
            
        def run_workflow():
            try:
                # 关键：直接将界面配置（agent_type, rag_enabled）注入工作流
                self._interactor.start_workflow_with_input(
                    workflow_input={
                        "query": text, 
                        "image_paths": files,
                        "agent_type": self.agent_type,
                        "rag_enabled": self.rag_enabled
                    }, 
                    workers=self.workers
                )
            except Exception as e:
                logging.error(f"Error starting workflow: {e}")
                
        threading.Thread(target=run_workflow, daemon=True).start()
        return history, gr.MultimodalTextbox(value=None, interactive=False)

    def start_interactor(self):
        with gr.Blocks(title="OmAgent Unified Web", css=self._custom_css) as demo:
            gr.Markdown("# 🤖 多模态 RAG 智能体 (OmAgent + RAG-Anything)")
            
            with gr.Tabs():
                with gr.Tab("💬 智能问答"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            agent_type_dd = gr.Dropdown(
                                choices=["CoT", "ToT", "ReAct", "React_Pro", "Reflexion", "PoT", "SC_CoT"],
                                value="CoT", label="推理引擎"
                            )
                            rag_toggle = gr.Checkbox(label="启用 RAG 增强", value=True)
                            
                            def update_config(a, r):
                                self.agent_type = a
                                self.rag_enabled = r
                                gr.Info(f"配置已更新：引擎={a}, RAG={'开启' if r else '关闭'}")
                            
                            agent_type_dd.change(update_config, [agent_type_dd, rag_toggle], None)
                            rag_toggle.change(update_config, [agent_type_dd, rag_toggle], None)
                            
                        with gr.Column(scale=4):
                            chatbot = gr.Chatbot(elem_id="OmAgent", type="messages", height=600)
                            chat_input = gr.MultimodalTextbox(interactive=True, file_count="multiple", placeholder="输入问题...", show_label=False)

                    chat_msg = chat_input.submit(self.add_message, [chatbot, chat_input], [chatbot, chat_input])
                    bot_msg = chat_msg.then(self.bot, chatbot, chatbot)
                    bot_msg.then(lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input])
                
                with gr.Tab("📚 知识库管理"):
                    with gr.Row():
                        file_input = gr.File(label="上传文档")
                        upload_btn = gr.Button("🚀 录入知识库", variant="primary")
                    upload_status = gr.Textbox(label="状态", interactive=False)
                    gr.Markdown("---")
                    refresh_btn = gr.Button("🔄 刷新列表")
                    doc_table = gr.Dataframe(headers=["文件名", "ID", "状态", "时间"], value=list_existing_docs_sync, interactive=False)
                    with gr.Row():
                        del_id = gr.Textbox(label="删除 ID", scale=3)
                        del_btn = gr.Button("🗑️ 删除", variant="stop", scale=1)
                    
                    upload_btn.click(upload_to_rag, file_input, upload_status).then(list_existing_docs, outputs=doc_table)
                    refresh_btn.click(list_existing_docs, outputs=doc_table)
                    del_btn.click(delete_doc, del_id, upload_status).then(list_existing_docs, outputs=doc_table)

                with gr.Tab("🕸️ 知识图谱"):
                    refresh_kg = gr.Button("🔄 刷新图谱")
                    kg_html = gr.HTML(get_kg_visualization_html)
                    refresh_kg.click(get_kg_visualization_html, outputs=kg_html)
            
            demo.launch()

# --- 工作流构建 (从 start_system_cli.py 移植) ---
def build_workflow():
    registry.import_module() 
    container.register_stm("RedisSTM")
    container.from_config(CURRENT_PATH.joinpath('container.yaml'))

    main_wf = ConductorWorkflow(name='universal_web_rag_system')

    # 在 Web 版中，我们跳过 SystemInputWorker 的 read_input 交互，直接从工作流输入获取
    # inputs={'query': main_wf.input('query'), 'agent_type': main_wf.input('agent_type'), ...}
    
    rag_task = simple_task(
        task_def_name='RAGRetrievalWorker', 
        task_reference_name='rag_retrieval',
        inputs={
            "query": main_wf.input('query'),
            "image_paths": main_wf.input('image_paths'),
            "rag_enabled": main_wf.input('rag_enabled'),
            "action": "query"
        }
    )

    context_task = simple_task(
        task_def_name='RAGContextWorker',
        task_reference_name='context_combiner',
        inputs={
            "query": main_wf.input('query'),
            "rag_res": rag_task.output('result'),
            "rag_enabled": main_wf.input('rag_enabled')
        }
    )

    final_query = context_task.output('final_query')

    # 定义各引擎链路 (与 CLI 版一致)
    cot_chain = [simple_task(task_def_name='CoTReasoning', task_reference_name='cot_reasoning', inputs={'query': final_query, 'cot_method': 'zero_shot'})]
    tot_chain = [
        simple_task(task_def_name='ThoughtDecomposition', task_reference_name='tot_decomp', inputs={'problem': final_query, 'requirements': 'General reasoning', 'qid': 'test'}),
        DoWhileTask(task_ref_name='tot_loop', tasks=[
            simple_task(task_def_name='ThoughtGenerator', task_reference_name='tot_gen'),
            simple_task(task_def_name='StateEvaluator', task_reference_name='tot_eval'),
            simple_task(task_def_name='SearchAlgorithm', task_reference_name='tot_search')
        ], termination_condition='if ($.tot_search["finish"] == true){false;} else {true;}')
    ]
    react_think = simple_task(task_def_name='ThinkAction', task_reference_name='react_think', inputs={'query': final_query})
    react_chain = [
        DoWhileTask(task_ref_name='react_loop', tasks=[
            react_think,
            simple_task(task_def_name='WikiSearch', task_reference_name='wiki_search', inputs={'action_output': react_think.output('action')})
        ], termination_condition='if ($.react_think["is_final"] == true){false;} else {true;}'),
        simple_task(task_def_name='ReactOutput', task_reference_name='react_out', inputs={'action_output': react_think.output('last_output')})
    ]
    rp_think = simple_task(task_def_name='Think', task_reference_name='rp_think', inputs={'query': final_query})
    rp_action = simple_task(task_def_name='Action', task_reference_name='rp_action', inputs={'thought': rp_think.output('thought')})
    rp_chain = [
        DoWhileTask(task_ref_name='rp_loop', tasks=[rp_think, rp_action], termination_condition='if ($.rp_action["is_final"] == true){false;} else {true;}'),
        simple_task(task_def_name='ReactOutput', task_reference_name='rp_out', inputs={'action_output': rp_action.output('output')})
    ]
    reflex_chain = [
        DoWhileTask(task_ref_name='reflex_loop', tasks=[
            simple_task(task_def_name='Think', task_reference_name='reflex_think', inputs={'query': final_query}),
            simple_task(task_def_name='Reflect', task_reference_name='reflex_reflect', inputs={'react_output': '${reflex_think.output.response}'})
        ], termination_condition='if ($.reflex_reflect["should_retry"] == false){false;} else {true;}')
    ]
    pot_exec = simple_task(task_def_name='PoTExecutor', task_reference_name='pot_exec', inputs={'query': final_query})
    pot_chain = [
        pot_exec,
        simple_task(task_def_name='ChoiceExtractor', task_reference_name='pot_extract', inputs={
            'query': final_query, 'prediction': pot_exec.output('last_output'),
            'completion_tokens': pot_exec.output('completion_tokens'), 'prompt_tokens': pot_exec.output('prompt_tokens')
        })
    ]
    sc_cot_chain = [simple_task(task_def_name='SCCoTReasoning', task_reference_name='sc_cot_reason', inputs={'query': final_query, 'id': 0})]

    agent_selector = SwitchTask(task_ref_name='agent_selector', case_expression=main_wf.input('agent_type'))
    for k, v in {"CoT": cot_chain, "ToT": tot_chain, "ReAct": react_chain, "React_Pro": rp_chain, "Reflexion": reflex_chain, "PoT": pot_chain, "SC_CoT": sc_cot_chain}.items():
        agent_selector.switch_case(k, v)
    agent_selector.default_case = cot_chain

    main_wf >> rag_task >> context_task >> agent_selector
    main_wf.register(overwrite=True)
    return main_wf

if __name__ == "__main__":
    workflow = build_workflow()
    config_path = CURRENT_PATH.joinpath('system_configs')
    local_workers = [
        system_workers.RAGRetrievalWorker(),
        system_workers.RAGContextWorker(),
        # 引擎相关的 Worker 会在运行中自动加载
    ]
    
    client = UnifiedWebpageClient(interactor=workflow, config_path=config_path, workers=local_workers)
    client.start_interactor()
