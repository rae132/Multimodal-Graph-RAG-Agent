from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry

@registry.register_worker()
class ReactOutput(BaseWorker):
    """Simple worker that passes through the final action output for React workflow"""
    
    def _run(self, action_output: any, *args, **kwargs):
        """Simply return the action output with any necessary state"""

        # state = self.stm(self.workflow_instance_id)
        state = self.stm(self.workflow_instance_id)
        query = state.get('query', '')
        id = state.get('id', '')
        token_usage = state.get('token_usage', {})
        context = state.get('context', '')

        if isinstance(action_output, dict):
            is_final = action_output.get('is_final', False)
            output_text = action_output.get('output', str(action_output))
        else:
            is_final = True if action_output and 'Finish[' in str(action_output) else False
            output_text = str(action_output)

        state['output'] = output_text
        return {
            'output': output_text,
            'context': context,
            'query': query,
            'id': id,
            'token_usage': token_usage,
            'is_final': is_final
        }
 