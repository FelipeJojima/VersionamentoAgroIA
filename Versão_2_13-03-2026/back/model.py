from functools import partial
from .GraphState import GraphState, get_model_from_cache ,retrieve_docs_similarity_node, create_model_node, grading_node, generate_answer_w_history_node, generate_answer_node, decide_to_generate,review_answer_node, modify_question_for_retry_to_mmr_node, modify_question_for_retry_to_threshold_node, retrieve_docs_similarity_threshold_node, retrieve_docs_mmr_node, modify_question_final_node
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

RAG_APP = None

# ==============================================================================
# DEFINIÇÕES 
# ==============================================================================
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large" 
DEFAULT_LLM_MODEL = "google/gemma-2-9b-it"
# FAISS_PATH = "back/bd/faiss_v1" #V1_all-mpnet-base-v2)
FAISS_PATH = "back/bd/faiss_index_com_chunking_intfloatmultilingual-e5-large"  #Bd com embedding model diferente com chunking (multilingual-e5-large) ==== V1_multilingual-e5-large
# FAISS_PATH = "back/bd/faiss_index_com_chunking,pior_que_o_outro-mpnet-base-v2" #Bd com chunking e all-mpnet-base-v2 ===== V3_all-mpnet-base-v2
# FAISS_PATH = "back/bd/faiss_index_sem_chunking-mpnet-base-v2" #Bd sem chunking, V2_all-mpnet-base-v2

embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# ==============================================================================
# CONSTRUÇÃO E EXECUÇÃO DO GRAFO
# ==============================================================================

def _constroeGrafo():
    """
    Função que constroi o grafo de execução do RAG
    """
    retrieve_docs_with_args = partial(retrieve_docs_similarity_node, embedding=embedder, faiss_path=FAISS_PATH, k_docs=20)
    retrieve_docs_with_args_sim_threshold = partial(retrieve_docs_similarity_threshold_node, embedding=embedder, faiss_path=FAISS_PATH, k_docs=20, threshold=0.7)
    retrieve_docs_with_args_mmr = partial(retrieve_docs_mmr_node, embedding=embedder, faiss_path=FAISS_PATH, k_docs=20, fetch_k=20,lambda_mult=0.5)

    workflow = StateGraph(GraphState)

    workflow.add_node("create_model", create_model_node)
    workflow.add_node("retrieve_docs_sim", retrieve_docs_with_args)
    workflow.add_node("grading", grading_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("adapt_the_question_sim_threshold", modify_question_for_retry_to_threshold_node)
    workflow.add_node("answer_only_chat_history", generate_answer_w_history_node)
    workflow.add_node("adapt_the_question_mmr" , modify_question_for_retry_to_mmr_node)
    workflow.add_node("adapt_final", modify_question_final_node)
    workflow.add_node("retrieve_docs_threshold", retrieve_docs_with_args_sim_threshold)
    workflow.add_node("retrieve_docs_mmr", retrieve_docs_with_args_mmr)


    workflow.add_edge(START, "create_model")
    workflow.add_edge("create_model", "retrieve_docs_sim")
    workflow.add_edge("retrieve_docs_sim", "grading")

    workflow.add_conditional_edges(
        "grading",
        decide_to_generate,
        {"generate": "generate_answer", "end": END, "gen_only_w_chat_history": "answer_only_chat_history"}
    )
    workflow.add_conditional_edges(
        "generate_answer",
        review_answer_node,
        {"sim": END, "Sim_Threshold": "adapt_the_question_sim_threshold", "MMR": "adapt_the_question_mmr", "não": "adapt_final"}
    )
    workflow.add_edge("answer_only_chat_history", END)
    workflow.add_edge("adapt_the_question_sim_threshold", "retrieve_docs_threshold")
    workflow.add_edge("retrieve_docs_threshold", "grading")
    workflow.add_edge("adapt_the_question_mmr", "retrieve_docs_mmr")
    workflow.add_edge("retrieve_docs_mmr", "grading")
    workflow.add_edge("adapt_final", "generate_answer")

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

def initialize_model():
    """
    Função pública para ser chamada UMA VEZ no início da aplicação Flask.
    Ela carrega o modelo na memória.
    """
    global RAG_APP
    if RAG_APP is None:

        print("Construindo o grafo e carregando o modelo... Isso pode levar alguns minutos.")
        
        RAG_APP = _constroeGrafo()
        print("Modelo e grafo carregados com sucesso!")

def search(pergunta: str, thread_id: str):
    """
    Função que envia a pergunta pro RAG e retorna a resposta resultante do RAG.
    """

    if RAG_APP is None:
        raise RuntimeError("O modelo não foi inicializado. Chame a função initialize_model() antes de usar a busca.")
    
    print(f"Processando pergunta para a thread: {thread_id}")
    
    initial_input = {
        "question": pergunta, 
        "model_id": DEFAULT_LLM_MODEL
    }

    config = {"configurable": {"thread_id": thread_id}}
    final_state = RAG_APP.invoke(initial_input, config)
    final_answer = final_state.get('answer', 'Nenhuma resposta foi gerada.')
    sources = final_state.get('sources', [])
    
    return {"answer": final_answer, "sources": sources}



