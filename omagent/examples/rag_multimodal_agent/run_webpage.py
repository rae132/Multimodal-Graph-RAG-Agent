# Import required modules and components
import os

# Bypass macOS OpenMP conflict caused by FAISS and other math libraries
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import asyncio
import logging as std_logging
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
import networkx as nx

# Setup standard logging
std_logging.basicConfig(level=std_logging.INFO)
logger = std_logging.getLogger(__name__)

CURRENT_PATH = Path(__file__).parent.absolute()

# CRITICAL FIX: Lock the RAG working directory to an absolute path 
# so knowledge base data is NEVER lost regardless of where the terminal is opened.
os.environ["RAG_WORKING_DIR"] = str(CURRENT_PATH / "rag_storage")

# Load environment variables from .env file
load_dotenv(CURRENT_PATH / ".env")

# Setup environment for OmAgent
os.environ["OMAGENT_MODE"] = "lite"

from omagent_core.clients.devices.webpage.lite_client import WebpageClient
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.utils.registry import registry

# Global RAG Tool instance (initialized lazily)
rag_tool = None

# --- Functions for UI ---

def get_rag_tool():
    global rag_tool
    if rag_tool is None:
        from omagent_core.tool_system.tools.rag_anything.rag_anything import RAGAnythingTool
        rag_tool = RAGAnythingTool()
    return rag_tool

async def upload_to_rag(file):
    if file is None:
        return "Please select a file first."
    
    file_path = file.name
    logger.info(f"UI: Uploading file to RAG: {file_path}")
    
    try:
        tool = get_rag_tool()
        result = await tool._arun(query="", action="upload", file_path=file_path)
        return result
    except Exception as e:
        import traceback
        logger.error(f"Upload failed: {traceback.format_exc()}")
        return f"Error: {str(e)}"

async def list_existing_docs():
    try:
        tool = get_rag_tool()
        docs = await tool._arun(action="list_docs")
        if not docs:
            return []
        return [[d["file_name"], d["doc_id"], d["status"], d["created_at"]] for d in docs]
    except Exception as e:
        logger.error(f"Failed to list docs: {e}")
        return []

def list_existing_docs_sync():
    """Synchronous wrapper for initial Gradio component value."""
    try:
        loop = asyncio.new_event_loop()
        return loop.run_until_complete(list_existing_docs())
    except Exception:
        return []

async def delete_doc(doc_id):
    if not doc_id:
        return "请在下方列表中选择或输入要删除的文档 ID"
    try:
        tool = get_rag_tool()
        res = await tool._arun(action="delete", doc_id=doc_id)
        return res
    except Exception as e:
        return f"删除失败: {e}"

def get_kg_visualization_html():
    kg_path = Path(os.environ.get("RAG_WORKING_DIR", "rag_storage")) / "graph_chunk_entity_relation.graphml"
    if not kg_path.exists():
        return """<div style='height: 500px; display: flex; flex-direction: column; align-items: center; justify-content: center; font-family: sans-serif; color: #666;'>
                    <h3>🕸️ 暂无知识图谱数据</h3>
                    <p>请先在「知识库管理」中上传并解析文档。</p>
                  </div>"""
    
    try:
        # Load the graph
        G = nx.read_graphml(kg_path)
        
        # Check if empty
        if G.number_of_nodes() == 0:
            return """<div style='height: 500px; display: flex; align-items: center; justify-content: center; font-family: sans-serif; color: #666;'>
                        <h3>图谱已生成，但没有提取到有效的实体和关系。</h3>
                      </div>"""

        # Convert to JSON securely
        nodes = []
        for node_id, data in G.nodes(data=True):
            # Clean string data for JS
            label = str(node_id).replace('"', "'")
            title = str(data.get("description", data)).replace('"', "'")
            nodes.append({
                "id": node_id, 
                "label": label[:15] + "..." if len(label) > 15 else label, 
                "title": title
            })
            
        edges = []
        for u, v, data in G.edges(data=True):
            label = str(data.get("relationship", data.get("weight", ""))).replace('"', "'")
            title = str(data.get("description", data)).replace('"', "'")
            edges.append({
                "from": u, 
                "to": v, 
                "label": label[:10] + "..." if len(label) > 10 else label, 
                "title": title
            })
            
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        
        # Robust HTML + JS inside an iframe to bypass Gradio JS stripping
        html_content = f"""
        <html>
        <head>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style type="text/css">
                body {{ margin: 0; padding: 0; background-color: #fcfcfc; }}
                #mynetwork {{ width: 100vw; height: 100vh; border: none; }}
            </style>
        </head>
        <body>
        <div id="mynetwork"></div>
        <script type="text/javascript">
            try {{
                var nodes = new vis.DataSet({nodes_json});
                var edges = new vis.DataSet({edges_json});
                var container = document.getElementById('mynetwork');
                
                var data = {{ nodes: nodes, edges: edges }};
                var options = {{
                    nodes: {{
                        shape: 'dot',
                        size: 20,
                        font: {{ size: 14, color: '#333' }},
                        borderWidth: 2,
                        color: {{ background: '#e0f2fe', border: '#1976d2', highlight: {{ background: '#bbdefb', border: '#1565c0' }} }}
                    }},
                    edges: {{
                        width: 1.5,
                        color: {{ color: '#9e9e9e', highlight: '#1976d2' }},
                        font: {{ size: 12, align: 'middle' }},
                        arrows: {{ to: {{ enabled: true, scaleFactor: 0.8 }} }}
                    }},
                    physics: {{
                        forceAtlas2Based: {{ gravitationalConstant: -50, centralGravity: 0.01, springLength: 100, springConstant: 0.08 }},
                        maxVelocity: 50,
                        solver: 'forceAtlas2Based',
                        timestep: 0.35,
                        stabilization: {{ iterations: 150 }}
                    }},
                    interaction: {{ hover: true, tooltipDelay: 200 }}
                }};
                
                var network = new vis.Network(container, data, options);
                
                network.once("stabilizationIterationsDone", function() {{
                    network.fit();
                }});
            }} catch (e) {{
                console.error("Error rendering graph:", e);
                document.getElementById('mynetwork').innerHTML = "<div style='padding:20px;color:red;'>渲染图谱时发生错误: " + e.message + "</div>";
            }}
        </script>
        </body>
        </html>
        """
        
        import html
        escaped_html = html.escape(html_content)
        return f'<iframe srcdoc="{escaped_html}" style="width: 100%; height: 600px; border: 1px solid #e0e0e0; border-radius: 8px;"></iframe>'
    except Exception as e:
        import traceback
        logger.error(f"KG Render Error: {traceback.format_exc()}")
        return f"<p style='color:red; font-family: monospace; white-space: pre-wrap;'>Error loading graph: {str(e)}</p>"

# --- Custom WebpageClient to override start_interactor ---
class CustomRAGWebpageClient(WebpageClient):
    def add_message(self, history, message):
        """Override add_message to bypass Redis queue and pass input directly to the workflow."""
        text = message.get("text", "")
        files = message.get("files", [])
        
        # Add to history for UI
        for x in files:
            history.append({"role": "user", "content": {"path": x}})
        if text:
            history.append({"role": "user", "content": text})
            
        import threading
        
        # Keep using the originally initialized workflow_instance_id so bot() matches worker's stream
        if self._workflow_instance_id is None:
            self._workflow_instance_id = self.workflow_instance_id
            
        def run_workflow():
            try:
                # Pass the query directly into the workflow input!
                self._interactor.start_workflow_with_input(
                    workflow_input={"query": text, "image_paths": files}, 
                    workers=self.workers
                )
            except Exception as e:
                logger.error(f"Error starting workflow: {e}")
                
        workflow_thread = threading.Thread(target=run_workflow, daemon=True)
        workflow_thread.start()
        
        return history, gr.MultimodalTextbox(value=None, interactive=False)

    def start_interactor(self):
        try:
            with gr.Blocks(title="Multimodal RAG Agent", css=self._custom_css) as demo:
                gr.Markdown("# 🤖 多模态 RAG 智能体 (OmAgent + RAG-Anything)")
                
                with gr.Tabs():
                    with gr.Tab("💬 智能问答"):
                        chatbot = gr.Chatbot(
                            elem_id="OmAgent",
                            type="messages",
                            height=600,
                        )

                        chat_input = gr.MultimodalTextbox(
                            interactive=True,
                            file_count="multiple",
                            placeholder="输入问题并按回车发送...",
                            show_label=False,
                        )

                        chat_msg = chat_input.submit(
                            self.add_message, [chatbot, chat_input], [chatbot, chat_input]
                        )
                        bot_msg = chat_msg.then(
                            self.bot, chatbot, chatbot, api_name="bot_response"
                        )
                        bot_msg.then(
                            lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]
                        )
                    
                    with gr.Tab("📚 知识库管理"):
                        gr.Markdown("### 1. 上传新文档 (PDF, TXT, Image)")
                        with gr.Row():
                            file_input = gr.File(label="选择文件", file_types=[".pdf", ".txt", ".docx", ".jpg", ".png"])
                            upload_btn = gr.Button("🚀 开始解析并录入", variant="primary")
                        upload_status = gr.Textbox(label="处理状态", interactive=False)

                        gr.Markdown("---")
                        gr.Markdown("### 2. 已有文档列表")
                        refresh_list_btn = gr.Button("🔄 刷新列表", variant="secondary", size="sm")
                        doc_table = gr.Dataframe(
                            headers=["文件名", "文档 ID", "状态", "录入时间"],
                            datatype=["str", "str", "str", "str"],
                            value=list_existing_docs_sync, # Use sync wrapper for init
                            interactive=False
                        )

                        with gr.Row():
                            delete_id_input = gr.Textbox(label="输入要删除的文档 ID", placeholder="从上方列表中复制 ID...", scale=3)
                            delete_btn = gr.Button("🗑️ 删除选中文档", variant="stop", scale=1)
                        delete_status = gr.Textbox(label="删除结果", interactive=False)

                        # Event handlers
                        upload_btn.click(fn=upload_to_rag, inputs=file_input, outputs=upload_status).then(
                            fn=list_existing_docs, outputs=doc_table
                        )
                        refresh_list_btn.click(fn=list_existing_docs, outputs=doc_table)
                        delete_btn.click(fn=delete_doc, inputs=delete_id_input, outputs=delete_status).then(
                            fn=list_existing_docs, outputs=doc_table
                        )

                    with gr.Tab("🕸️ 知识图谱展示"):
                        gr.Markdown("### 领域知识实体关系图 (基于 LightRAG 自动提取)")
                        refresh_kg_btn = gr.Button("🔄 刷新图谱", variant="secondary")
                        kg_display = gr.HTML(get_kg_visualization_html)
                        # When tab is clicked, or button is clicked, refresh
                        refresh_kg_btn.click(fn=get_kg_visualization_html, outputs=kg_display)
                
                demo.launch()
        except KeyboardInterrupt:
            logger.info("\nDetected Ctrl+C, stopping workflow...")
            if self._workflow_instance_id is not None:
                self._interactor._executor.terminate(
                    workflow_id=self._workflow_instance_id
                )
            raise

# --- Main Entry ---
if __name__ == "__main__":
    CURRENT_PATH = Path(__file__).parent
    registry.import_module(project_path=CURRENT_PATH.joinpath("agent"))

    from agent.simple_vqa.simple_vqa import RAGMultimodalWorker

    workflow = ConductorWorkflow(name='rag_multimodal_agent_workflow')
    task = simple_task(
        task_def_name='RAGMultimodalWorker', 
        task_reference_name='rag_multimodal_task', 
        inputs={'query': workflow.input('query'), 'image_paths': workflow.input('image_paths')}
    )
    workflow.add(task)

    config_path = CURRENT_PATH.joinpath("configs")
    agent_client = CustomRAGWebpageClient(
        interactor=workflow, config_path=config_path, workers=[]
    )
    agent_client.start_interactor()
