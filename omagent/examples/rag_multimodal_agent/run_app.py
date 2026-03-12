# Import required modules and components
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup environment for OmAgent
os.environ["OMAGENT_MODE"] = "lite"

from omagent_core.clients.devices.app.client import AppClient
from omagent_core.engine.workflow.conductor_workflow import ConductorWorkflow
from omagent_core.utils.registry import registry
from omagent_core.utils.logger import logging

# Import project-specific workers
from agent.input_interface.input_interface import InputInterface
from agent.simple_vqa.simple_vqa import RAGMultimodalWorker

if __name__ == "__main__":
    CURRENT_PATH = Path(__file__).parent
    registry.import_module(project_path=CURRENT_PATH)

    workflow = ConductorWorkflow(name='rag_multimodal_agent_workflow')
    # Add main multimodal task
    workflow.add_task(worker_name='RAGMultimodalWorker', task_reference_name='rag_multimodal_task', 
                      input_params={'query': '${workflow.input.query}', 'image_paths': '${workflow.input.image_paths}'})

    config_path = CURRENT_PATH.joinpath("configs")
    agent_client = AppClient(
        interactor=workflow, config_path=config_path, workers=[InputInterface()]
    )
    agent_client.start_interactor()
