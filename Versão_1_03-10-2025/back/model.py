# back/model.py

# --- Importações ---
from functools import partial
from langgraph.checkpoint.memory import MemorySaver
from .GraphState import GraphState,gen_answer_fault,reset_answer, get_model_from_cache ,retrieve_docs_similarity_node, create_model_node, grading_node, generate_answer_node, decide_to_generate,review_answer_node, retrieve_docs_similarity_threshold_node, retrieve_docs_mmr_node
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph

# --- Constantes e Configurações ---
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
FAISS_PATH = "back/bd/faiss_index_com_chunking_intfloatmultilingual-e5-large"
DEFAULT_LLM_MODEL = "google/gemma-2-9b-it"

embedder = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={'device': 'cpu'}
)

RAG_APP = None

# ==============================================================================
# CONSTRUÇÃO DO GRAFO 
# ==============================================================================

def _build_graph():
    """
    Constrói e compila o grafo de execução do RAG, mas SEM o checkpointer.
    O checkpointer será adicionado dinamicamente depois.
    """
    k_docs=15
    retrieve_docs_with_args = partial(retrieve_docs_similarity_node, embedding=embedder, faiss_path=FAISS_PATH, k_docs=k_docs)
    retrieve_docs_with_args_sim_threshold = partial(retrieve_docs_similarity_threshold_node, embedding=embedder, faiss_path=FAISS_PATH, k_docs=k_docs, threshold=0.7)
    retrieve_docs_with_args_mmr = partial(retrieve_docs_mmr_node, embedding=embedder, faiss_path=FAISS_PATH, k_docs=k_docs, fetch_k=20,lambda_mult=0.5)

    workflow = StateGraph(GraphState)

    workflow.add_node("create_model", create_model_node)
    workflow.add_node("retrieve_docs_sim", retrieve_docs_with_args)
    workflow.add_node("grading", grading_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("retrieve_docs_threshold", retrieve_docs_with_args_sim_threshold)
    workflow.add_node("retrieve_docs_mmr", retrieve_docs_with_args_mmr)
    workflow.add_node("reset_answer_field", reset_answer)
    workflow.add_node("gen_response_fault", gen_answer_fault)

    workflow.add_edge(START, "create_model")
    workflow.add_edge("create_model", "reset_answer_field")
    workflow.add_edge("reset_answer_field", "retrieve_docs_sim")
    workflow.add_edge("retrieve_docs_sim", "grading")

    workflow.add_conditional_edges(
        "grading",
        decide_to_generate,
        {"generate": "generate_answer", "no_docs_try_threshold": "retrieve_docs_threshold", "no_docs_try_mmr": "retrieve_docs_mmr", "end": END, "no": "gen_response_fault"}
    )
    workflow.add_conditional_edges(
        "generate_answer",
        review_answer_node,
        {"sim": END, "Sim_Threshold": "retrieve_docs_threshold", "MMR": "retrieve_docs_mmr", "não": "gen_response_fault", "end": END}
    )
    workflow.add_edge("retrieve_docs_threshold", "grading")
    workflow.add_edge("retrieve_docs_mmr", "grading")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

# ==============================================================================
# INICIALIZAÇÃO E EXECUÇÃO
# ==============================================================================
def initialize_model():
    """
    Função pública para ser chamada UMA VEZ.
    """
    global RAG_APP
    if RAG_APP is None:
        print("Construindo o grafo e carregando o modelo...")
        RAG_APP = _build_graph()
        print("Modelo e grafo carregados com sucesso!")

def search(pergunta: str, thread_id: str):
    """
    Função que envia a pergunta pro RAG e retorna a resposta.
    O histórico é gerenciado na memória pelo MemorySaver.
    """
    if RAG_APP is None:
        raise RuntimeError("O modelo não foi inicializado.")
    
    print(f"Processando pergunta para a thread: {thread_id}")
    
    initial_input = {
        "question": pergunta,
        "model_id": DEFAULT_LLM_MODEL
    }

    config = {"configurable": {"thread_id": thread_id},"recursion_limit": 100}
    
    final_state = RAG_APP.invoke(initial_input, config)
    
    final_answer = final_state.get('answer', 'Nenhuma resposta foi gerada.')
    sources = final_state.get("sources", [])
    
    return {"answer": final_answer, "sources": sources}