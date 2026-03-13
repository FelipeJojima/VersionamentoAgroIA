from functools import partial
from GraphState_testes import GraphState,gen_answer_fault,reset_answer, monitor_recursos ,retrieve_docs_similarity_node, create_model_node, grading_node, generate_answer_node, decide_to_generate,review_answer_node, retrieve_docs_similarity_threshold_node, retrieve_docs_mmr_node
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from IPython.display import Image
from langchain_community.vectorstores import FAISS

RAG_APP = None

# ==============================================================================
# DEFINIÇÕES 
# ==============================================================================
# EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
# EMBEDDING_MODEL = "intfloat/multilingual-e5-large" 
# DEFAULT_LLM_MODEL = "google/gemma-2-9b-it"
# DEFAULT_LLM_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# FAISS_PATH = "back/bd/faiss_v1" #V1_all-mpnet-base-v2)
# FAISS_PATH = "back/bd/faiss_index_com_chunking_intfloatmultilingual-e5-large"  #Bd com embedding model diferente com chunking (multilingual-e5-large) ==== V1_multilingual-e5-large
# FAISS_PATH = "back/bd/faiss_index_com_chunking,pior_que_o_outro-mpnet-base-v2" #Bd com chunking e all-mpnet-base-v2 ===== V3_all-mpnet-base-v2
# FAISS_PATH = "back/bd/faiss_index_sem_chunking-mpnet-base-v2" #Bd sem chunking, V2_all-mpnet-base-v2


# ==============================================================================
# CONSTRUÇÃO E EXECUÇÃO DO GRAFO
# ==============================================================================

@monitor_recursos
def _constroeGrafo(switch_faiss: int, k_docs: int):
    """
    Função que constroi o grafo de execução do RAG
    """
    match switch_faiss:
        case 1:
            FAISS_PATH = "bd/faiss_v1" #V1_all-mpnet-base-v2)
            EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
        case 2:
            FAISS_PATH = "bd/faiss_index_com_chunking,pior_que_o_outro-mpnet-base-v2" #Bd com chunking e all-mpnet-base-v2 ===== V3_all-mpnet-base-v2
            EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
        case 3:
            FAISS_PATH = "bd/faiss_index_sem_chunking-mpnet-base-v2" #Bd sem chunking, V2_all-mpnet-base-v2
            EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
        case 4:
            EMBEDDING_MODEL = "intfloat/multilingual-e5-large" 
            FAISS_PATH = "bd/faiss_index_com_chunking_intfloatmultilingual-e5-large"  #Bd com embedding model diferente com chunking (multilingual-e5-large) ==== V1_multilingual-e5-large
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device':'cpu'})
    try:
        vectorstore = FAISS.load_local(FAISS_PATH, embeddings=embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o índice FAISS de '{FAISS_PATH}'. {e}")
    retrieve_docs_with_args = partial(retrieve_docs_similarity_node,vectorstore=vectorstore, k_docs=k_docs,fetch_k=50)
    retrieve_docs_with_args_sim_threshold = partial(retrieve_docs_similarity_threshold_node,vectorstore=vectorstore, k_docs=k_docs, threshold=0.3,fetch_k=50)
    retrieve_docs_with_args_mmr = partial(retrieve_docs_mmr_node, vectorstore=vectorstore, k_docs=k_docs, fetch_k=50,lambda_mult=0.5)


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

    # workflowTestesFAISS = StateGraph(GraphState)
    # workflowTestesFAISS.add_node("create", create_model_node)
    # workflowTestesFAISS.add_node("sim", retrieve_docs_with_args)
    # workflowTestesFAISS.add_node("thr", retrieve_docs_with_args_sim_threshold)
    # workflowTestesFAISS.add_node("mmr", retrieve_docs_with_args_mmr)
    # workflowTestesFAISS.add_edge(START, "create")
    # workflowTestesFAISS.add_edge("create", "sim")
    # workflowTestesFAISS.add_edge("sim", "thr")
    # workflowTestesFAISS.add_edge("thr", "mmr")
    # workflowTestesFAISS.add_edge("mmr", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

@monitor_recursos
def initialize_model(emb:int, k_docs:int):
    """
    Função pública para ser chamada UMA VEZ no início da aplicação Flask.
    Ela carrega o modelo na memória.
    """
    global RAG_APP
    if RAG_APP is None:

        # print("Construindo o grafo e carregando o modelo... Isso pode levar alguns minutos.")
        
        RAG_APP = _constroeGrafo(emb,k_docs)
        
        # print("Modelo e grafo carregados com sucesso!")
        try:
            img_data = RAG_APP.get_graph().draw_mermaid_png()
            with open("graph_final.png", "wb") as f:
                f.write(img_data)
            Image(img_data)
            #print("Grafo visual salvo como 'graph_final.png'")
        except Exception as e:
            print(f"Não foi possível gerar a imagem do grafo: {e}")

@monitor_recursos
def search(pergunta: str, thread_id: str, model:int):
    """
    Função que envia a pergunta pro RAG e retorna a resposta resultante do RAG.
    """

    if RAG_APP is None:
        raise RuntimeError("O modelo não foi inicializado. Chame a função initialize_model() antes de usar a busca.")
    
    # print(f"Processando pergunta para a thread: {thread_id}")
    match model:
        case 1: 
            DEFAULT_LLM_MODEL = "google/gemma-2-9b-it"
        case 2:
            DEFAULT_LLM_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    initial_input = {
        "question": pergunta, 
        "model_id": DEFAULT_LLM_MODEL
    }

    config = {"configurable": {"thread_id": thread_id},"recursion_limit": 100}
    final_state = RAG_APP.invoke(initial_input, config)
    final_answer = "\n"
    final_answer += final_state.get('answer', 'Nenhuma resposta foi gerada.')
    final_answer += "\n"
    source = final_state.get('sources', [])
    sources = ""
    for s in source:
        sources += "\n"
        sources += s.get('url')
        sources += "\n"
    histories = final_state.get('chat_history', None)
    history = ""
    for i, h in enumerate(histories):
        history += f"\n{i} Pergunta\n"
        history += h.content
        history += "\n"
    s_method = final_state.get('search_method', None)
    
    return {"answer": final_answer, "sources": sources, "history": history, "search_method": s_method, "question": pergunta}


# initialize_model(4,40)
# search("Cite os principais cultivares da banana recomendados para o estado de Santa Catarina.", 123123, 1)
