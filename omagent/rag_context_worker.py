from omagent_core.engine.worker.base import BaseWorker
from omagent_core.utils.registry import registry

@registry.register_worker()
class RAGContextWorker(BaseWorker):
    def _run(self, query: str, rag_res: str = "", rag_enabled: bool = False, *args, **kwargs):
        if not rag_enabled or not rag_res:
            return {"final_query": query}
        
        # Prepend RAG context to the query
        final_query = f"""Below is some context from the RAG knowledge base:
{rag_res}

Based on the context above, please answer the user's question:
{query}
"""
        return {"final_query": final_query}
