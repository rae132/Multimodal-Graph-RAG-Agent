import os
from pathlib import Path
from dotenv import load_dotenv

# 1. 初始化
load_dotenv()
os.environ["OMAGENT_MODE"] = "lite"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from omagent_core.utils.container import container
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.engine.workflow.task.simple_task import simple_task
from omagent_core.engine.workflow.task.switch_task import SwitchTask
from omagent_core.engine.workflow.task.do_while_task import DoWhileTask
from omagent_core.utils.registry import registry
from omagent_core.clients.devices.cli import DefaultClient
from omagent_core.utils.logger import logging

import system_workers

# 屏蔽 asyncio 和 lightrag 的清理期报错日志
import logging as std_logging
std_logging.getLogger("asyncio").setLevel(std_logging.CRITICAL)
std_logging.getLogger("lightrag").setLevel(std_logging.ERROR)

logging.init_logger("omagent", "omagent", level="INFO")

def start_system():
    CURRENT_PATH = Path(__file__).parent
    
    # 2. 核心组件注册
    registry.import_module() 
    container.register_stm("RedisSTM")
    container.from_config(CURRENT_PATH.joinpath('container.yaml'))

    # 3. 编排主逻辑
    main_wf = ConductorWorkflow(name='universal_rag_system')

    # Step 0: 系统主菜单
    menu_task = simple_task(task_def_name='SystemMenuWorker', task_reference_name='sys_menu')

    # ========================== 分支 A: 知识管理 (RAG Upload) ==========================
    upload_input_task = simple_task(task_def_name='KnowledgeUploadWorker', task_reference_name='upload_input')
    upload_exec_task = simple_task(
        task_def_name='RAGRetrievalWorker',
        task_reference_name='upload_exec',
        inputs={
            'action': 'upload',
            'file_path': upload_input_task.output('file_path')
        }
    )
    knowledge_path = [upload_input_task, upload_exec_task]

    # ========================== 分支 B: 智能对话 (Chat with Agents) ==========================
    input_task = simple_task(task_def_name='SystemInputWorker', task_reference_name='sys_input')
    
    rag_task = simple_task(
        task_def_name='RAGRetrievalWorker', 
        task_reference_name='rag_retrieval',
        inputs={
            "query": input_task.output('query'),
            "image_paths": input_task.output('image_paths'),
            "rag_enabled": input_task.output('rag_enabled'),
            "action": "query"
        }
    )

    context_task = simple_task(
        task_def_name='RAGContextWorker',
        task_reference_name='context_combiner',
        inputs={
            "query": input_task.output('query'),
            "rag_res": rag_task.output('result'),
            "rag_enabled": input_task.output('rag_enabled')
        }
    )

    final_query = context_task.output('final_query')

    # 引擎定义
    cot_chain = [simple_task(task_def_name='CoTReasoning', task_reference_name='cot_reasoning',
                             inputs={'query': final_query, 'cot_method': 'zero_shot'})]

    tot_decomp = simple_task(task_def_name='ThoughtDecomposition', task_reference_name='tot_decomp',
                             inputs={'problem': final_query, 'requirements': 'General reasoning', 'qid': 'test'})
    tot_loop = DoWhileTask(
        task_ref_name='tot_loop',
        tasks=[
            simple_task(task_def_name='ThoughtGenerator', task_reference_name='tot_gen'),
            simple_task(task_def_name='StateEvaluator', task_reference_name='tot_eval'),
            simple_task(task_def_name='SearchAlgorithm', task_reference_name='tot_search')
        ],
        termination_condition='if ($.tot_search["finish"] == true){false;} else {true;}'
    )
    tot_chain = [tot_decomp, tot_loop]

    react_think = simple_task(task_def_name='ThinkAction', task_reference_name='react_think', inputs={'query': final_query})
    react_loop = DoWhileTask(
        task_ref_name='react_loop',
        tasks=[
            react_think,
            simple_task(task_def_name='WikiSearch', task_reference_name='wiki_search', inputs={'action_output': react_think.output('action')})
        ],
        termination_condition='if ($.react_think["is_final"] == true){false;} else {true;}'
    )
    react_chain = [react_loop, simple_task(task_def_name='ReactOutput', task_reference_name='react_out', inputs={'action_output': react_think.output('last_output')})]

    rp_think = simple_task(task_def_name='Think', task_reference_name='rp_think', inputs={'query': final_query})
    rp_action = simple_task(task_def_name='Action', task_reference_name='rp_action', inputs={'thought': rp_think.output('thought')})
    rp_loop = DoWhileTask(
        task_ref_name='rp_loop',
        tasks=[rp_think, rp_action],
        termination_condition='if ($.rp_action["is_final"] == true){false;} else {true;}'
    )
    rp_chain = [rp_loop, simple_task(task_def_name='ReactOutput', task_reference_name='rp_out', inputs={'action_output': rp_action.output('output')})]

    reflex_think = simple_task(task_def_name='Think', task_reference_name='reflex_think', inputs={'query': final_query})
    reflex_reflect = simple_task(task_def_name='Reflect', task_reference_name='reflex_reflect', inputs={'react_output': reflex_think.output('response')})
    reflex_loop = DoWhileTask(
        task_ref_name='reflex_loop',
        tasks=[reflex_think, reflex_reflect],
        termination_condition='if ($.reflex_reflect["should_retry"] == false){false;} else {true;}'
    )
    reflex_chain = [reflex_loop]

    pot_exec = simple_task(task_def_name='PoTExecutor', task_reference_name='pot_exec', inputs={'query': final_query})
    pot_chain = [
        pot_exec,
        simple_task(task_def_name='ChoiceExtractor', task_reference_name='pot_extract', 
                    inputs={
                        'query': final_query, 
                        'prediction': pot_exec.output('last_output'),
                        'completion_tokens': pot_exec.output('completion_tokens'),
                        'prompt_tokens': pot_exec.output('prompt_tokens')
                    })
    ]

    sc_cot_chain = [simple_task(task_def_name='SCCoTReasoning', task_reference_name='sc_cot_reason', 
                                inputs={'query': final_query, 'id': 0})]

    agent_selector = SwitchTask(task_ref_name='agent_selector', case_expression=input_task.output('agent_type'))
    agent_selector.switch_case("CoT", cot_chain)
    agent_selector.switch_case("ToT", tot_chain)
    agent_selector.switch_case("ReAct", react_chain)
    agent_selector.switch_case("React_Pro", rp_chain)
    agent_selector.switch_case("Reflexion", reflex_chain)
    agent_selector.switch_case("PoT", pot_chain)
    agent_selector.switch_case("SC_CoT", sc_cot_chain)
    agent_selector.default_case = cot_chain

    chat_path = [input_task, rag_task, context_task, agent_selector]

    # ========================== 主流程 Switch ==========================
    main_selector = SwitchTask(task_ref_name='main_selector', case_expression=menu_task.output('action_type'))
    main_selector.switch_case("chat", chat_path)
    main_selector.switch_case("knowledge", knowledge_path)

    # 工作流连线
    main_wf >> menu_task >> main_selector

    main_wf.register(overwrite=True)
    
    # 4. 启动客户端
    config_path = CURRENT_PATH.joinpath('system_configs')
    
    local_workers = [
        system_workers.SystemMenuWorker(),
        system_workers.KnowledgeUploadWorker(),
        system_workers.SystemInputWorker(),
        system_workers.RAGRetrievalWorker(),
        system_workers.RAGContextWorker(),
        system_workers.InputInterface()
    ]
    
    logging.info(f"Starting unified agent system with RAG Management...")

    client = DefaultClient(
        interactor=main_wf, 
        config_path=config_path, 
        workers=local_workers
    )
    client.start_interactor()

if __name__ == "__main__":
    start_system()
