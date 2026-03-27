
from pydantic import BaseModel, Field
import torch
from typing import List, Any, Dict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from back.extract_functions import _internal_extract_resposta_gemma, _internal_extrair_descricao, _internal_extract_resposta_llama
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from back.prompts import PROMPT_ANSWER, PROMPT_ANSWER_FAULT, PROMPT_RESUMO_HISTORICO, PROMPT_AVALIAR_CONTEXTO, PROMPT_ANSWER_WITHOUT_HISTORY, PROMPT_QUERY_ASSUNTO,PROMPT_QUERY_OBJETIVO, PROMPT_QUERY_CONTEUDO_RESPOSTA, PROMPT_QUERY_INFORMACOES_RELEVANTES_SOBRE
from ragas import EvaluationDataset, evaluate, RunConfig
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference,ResponseGroundedness
from ragas.metrics import ContextRelevance 
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.embeddings.base import LangchainEmbeddingsWrapper
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

class GraphState(BaseModel):
    question: str
    model_id: str
    answer: str = ""
    search_method: str = "similarity"
    documents: List[Document] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    chat_history: List[BaseMessage] = Field(default_factory=list)
    gen_lock: bool = False


_model_cache = {}


def formatar_historico(hist: List[BaseMessage], max_messages: int) -> str:
    count = 0
    final_hist =[]
    if not hist:
        return "Nenhum histórico de conversa ainda."
    for c in hist:
        if count > max_messages:
            break
        final_hist.append(c)
        count += 1
    formatted_lines = []
    for msg in final_hist:
        if isinstance(msg, HumanMessage):
            formatted_lines.append(f"Usuário: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted_lines.append(f"Sua Resposta Anterior: {msg.content}")
    return "\n".join(formatted_lines)





def get_model_from_cache_gemma(model_id: str):
    """
    Verifica o cache. Se o modelo não estiver lá, ele o carrega,
    armazena no cache e o retorna.
    """
    global _model_cache
    if model_id not in _model_cache:
        model = OllamaLLM(model=model_id, temperature=0.25,top_k=10,top_p=0.7,repeat_penalty=1.35, repeat_last_n=32)
        _model_cache[model_id] = model
        
    return _model_cache[model_id]

def get_model_from_cache_llama(model_id: str):
    """
    Verifica o cache. Se o modelo não estiver lá, ele o carrega,
    armazena no cache e o retorna.
    """
    global _model_cache
    if model_id not in _model_cache:
        model = OllamaLLM(model=model_id, temperature=0.25,top_k=10,top_p=0.7,repeat_penalty=1.35,repeat_last_n=32)
        _model_cache[model_id] = model

    return _model_cache[model_id]


def get_model_from_cache(model_id:str):
    if model_id=="mygemma3:12b":
        return get_model_from_cache_gemma(model_id)
    elif model_id=="myllama32-vision:11b":
        return get_model_from_cache_llama(model_id)
    
def create_model_node(state: GraphState) -> None:
    """
    Este nó age como um "pré-aquecedor". Ele garante que o modelo
    especificado no estado (`model_id`) esteja carregado no cache.
    """
    model_id = state.model_id
    if not model_id:
        raise ValueError("model_id não foi fornecido no estado inicial do grafo.")
    get_model_from_cache(model_id)
    return

def generate_new_query(prompt: str, model_id: str, original_question: str) -> str:
    print("\n\n\n\nGenerating new sub-query.....\n")
    chat = get_model_from_cache(model_id=model_id)

    prompt_template = ChatPromptTemplate.from_template(prompt)

    chain = prompt_template.pipe(chat).pipe(StrOutputParser())
    with torch.no_grad():
        resp = chain.invoke({
            "pergunta": original_question
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if model_id=="mygemma3:12b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif model_id=="myllama32-vision:11b":
        fResp = _internal_extract_resposta_llama(resp)

    return fResp


def retrieve_docs_similarity_node(state: GraphState,vectorstore: FAISS,  EMBEDDING_MODEL: str, k_docs: int = 5, fetch_k: int = 30) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método similarity_search_by_vector() para a busca.
    """
    print("Retrieving...\n")
    search_query = state.question

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device':'cpu'})
    q_vector = embedder.embed_query(search_query)
    new_queries = []
    prompts_queries = [PROMPT_QUERY_ASSUNTO,PROMPT_QUERY_OBJETIVO, PROMPT_QUERY_CONTEUDO_RESPOSTA, PROMPT_QUERY_INFORMACOES_RELEVANTES_SOBRE]
    for prompt in prompts_queries:
        new_queries.append(generate_new_query(prompt=prompt, model_id=state.model_id,original_question=search_query))


    all_queries = [search_query] + new_queries 

    all_retrieved_docs = []

    q_vectors = []
    for query in all_queries:
        q_vectors.append(embedder.embed_query(query))
    num_docs_question = k_docs // 2
    num_docs_per_query = (k_docs-num_docs_question) // 4
    for i,q_vector in enumerate(q_vectors):
        if i==0:
            retrieved_docs = vectorstore.similarity_search_by_vector(
                embedding=q_vector,
                k=num_docs_question,      
                fetch_k=fetch_k
            )
            all_retrieved_docs.extend(retrieved_docs)

            q_vector = []    
        else:
            retrieved_docs = vectorstore.similarity_search_by_vector(
                embedding=q_vector,
                k=num_docs_per_query,      
                fetch_k=fetch_k
            )
            all_retrieved_docs.extend(retrieved_docs)

            q_vector = []


    unique_docs = {}
    for doc in all_retrieved_docs:
        unique_docs[doc[0].page_content] = doc[0]

    docs = list(unique_docs.values())
    proc_docs = []
    for doc in docs:
        titulo = doc[0].metadata.get("titulo", "Sem título")
        arquivo = doc[0].metadata.get("arquivo_origem", "")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc[0].page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"url": url, "title": titulo}))
    new_history = state.chat_history
    if len(new_history)> 2:
        if isinstance(new_history[len(new_history)-1], AIMessage) and new_history[len(new_history)-2].content != search_query:
            new_history.append(HumanMessage(search_query))
        else:  
            if search_query!=new_history[len(new_history)-1].content:
                new_history.append(HumanMessage(search_query))
    else:
        new_history.append(HumanMessage(search_query))
    return {"documents": proc_docs, "chat_history": new_history, "search_method": "Similarity"}

def grading_node(state: GraphState, EMBEDDING_MODEL:str) -> Dict[str, Any]:
    """
    Nó que avalia os documentos recuperados do FAISS. Retorna os documentos que o modelo avaliar relevante para a pergunta do usuário.
    """
    print("Grading...\n")
    model_id = state.model_id
    chat = get_model_from_cache(model_id)
    question = state.question
    docs = state.documents

    if not docs:
        return {"documents": [], "sources": []}

    rel_docs = []
    sources = []

    embedder_hf = HuggingFaceEmbeddings(model=EMBEDDING_MODEL,model_kwargs={ 'device':'cpu'})
    embedder = LangchainEmbeddingsWrapper(embeddings=embedder_hf)
    data = []

    

    llm_ragas = get_model_from_cache(model_id=model_id)
    
    evaluator_llm = LangchainLLMWrapper(langchain_llm=llm_ragas)
    runconfig = RunConfig(max_workers=1)
    for d in docs: 
        data.append(
            {"user_input": question,
            "retrieved_contexts": [d.page_content]
            }
        )
        eval_data = EvaluationDataset.from_list(data)
        try:
            results = evaluate(
                dataset=eval_data, 
                metrics=[
                    ContextRelevance(llm=evaluator_llm)
                ],
                llm=evaluator_llm, 
                embeddings=embedder, 
                run_config=runconfig,
                show_progress=False
            )
        except Exception as e:
            results = {}
        data.clear()
        if((results['nv_context_relevance'][0])>=0.5):
            rel_docs.append(d)
            print(d.metadata)
            sources.append(d.metadata)

    return {"documents": rel_docs, "sources": sources}

def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """
    Nó que gera a resposta para a pergunta a partir dos documentos e do histórico do chat.
    """
    print("Generating...\n")
    model_id = state.model_id
    chat = get_model_from_cache(model_id)

    question = state.question
    docs = state.documents
    chat_history = state.chat_history
    
    all_content = "\n\n---\n\n".join([d.page_content for d in docs])
    

    
    formatted_history = formatar_historico(chat_history, 4)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_ANSWER)
    
    chain = prompt_template.pipe(chat).pipe(StrOutputParser())
    with torch.no_grad():
        resp = chain.invoke({
            "pergunta": question, 
            "contexto": all_content,
            "historico": formatted_history
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if state.model_id=="mygemma3:12b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision:11b":
        fResp = _internal_extract_resposta_llama(resp)
    new_history = state.chat_history
    if isinstance(new_history[len(new_history)-1], AIMessage):
        new_history.pop(len(new_history)-1)
        new_history.append(AIMessage(fResp))
    else:
        new_history.append(AIMessage(fResp))
    return {"answer": fResp, "chat_history": new_history}

def review_answer_node(state: GraphState, EMBEDDING_MODEL: str) -> str:
    """
    Nó que avalia a resposta, verificando se a questão do usuário foi respondida.
    """
    print("Reviewing...\n")
    query = state.question
    response = state.answer
    documents = state.documents
    embedder_hf = HuggingFaceEmbeddings(model=EMBEDDING_MODEL,model_kwargs={ 'device':'cpu'})
    embedder = LangchainEmbeddingsWrapper(embeddings=embedder_hf)
    data = []
    data.append(
        {"user_input": query,
         "retrieved_contexts": [doc.page_content for doc in documents],
         "response": response}
    )
    eval_data = EvaluationDataset.from_list(data)

    llm_ragas = get_model_from_cache(model_id=state.model_id)
    
    evaluator_llm = LangchainLLMWrapper(langchain_llm=llm_ragas)
    runconfig = RunConfig(max_workers=1)

    try:
        results = evaluate(
            dataset=eval_data, 
            metrics=[
                LLMContextPrecisionWithoutReference(llm=evaluator_llm), 
                ResponseGroundedness(llm=evaluator_llm), 
            ],
            llm=evaluator_llm, 
            embeddings=embedder, 
            run_config=runconfig,
            show_progress=False
        )
    except Exception as e:
        results = {}
    score = 0.0  
    
    try:
        score_list = results['llm_context_precision_without_reference']
        
        if score_list and len(score_list) > 0:
            score_value = score_list[0]
            
            if score_value is not None and score_value == score_value:
                score = score_value
                
    except KeyError:
        print("    -> Aviso RAGAs: Métrica 'llm_context_precision_without_reference' falhou e não foi encontrada.")
    except Exception as e:
        print(f"    -> Aviso RAGAs: Erro inesperado ao ler o score: {e}")

    
    if score < 0.75:
        if state.search_method == "mmr":
            if state.answer=="":
                return "não"
            else:
                return "end"
        if state.search_method == "Similarity":
            return "MMR"
        return "Similarity"
    else:
        return "sim"


def review_state(state:GraphState):
    return

def decide_to_generate(state: GraphState) -> str:
    """
    Decide o próximo passo: gerar com documentos, gerar apenas com histórico, ou terminar.
    """
    if state.documents:
        return "generate"
    else:
        if state.search_method=="Similarity":
            return "no_docs_try_mmr"
        elif state.search_method=="similarity_threshold":
            return "no_docs_try_sim"
        else:
            if state.answer=="":
                return "no"
            else:
                return "end"

def gen_answer_fault(state: GraphState):
    print("Generating...\n")
    model_id = state.model_id
    chat = get_model_from_cache(model_id)

    question = state.question

    prompt_template = ChatPromptTemplate.from_template(PROMPT_ANSWER_FAULT)
    
    chain = prompt_template.pipe(chat).pipe(StrOutputParser())
    with torch.no_grad():
        resp = chain.invoke({
            "pergunta": question, 
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if state.model_id=="mygemma3:12b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision:11b":
        fResp = _internal_extract_resposta_llama(resp)
    new_history = state.chat_history
    new_history.append(AIMessage(fResp))
    return {"answer": fResp, "chat_history": new_history}



def retrieve_docs_similarity_threshold_node(state: GraphState, vectorstore: FAISS, EMBEDDING_MODEL: str, threshold: float = 0.5,  k_docs: int = 4,fetch_k: int = 30) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    print("Retrieving...\n")
    search_query = state.question
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device':'cpu'})

    q_vector = embedder.embed_query(search_query)
    new_queries = []
    prompts_queries = [PROMPT_QUERY_ASSUNTO,PROMPT_QUERY_OBJETIVO, PROMPT_QUERY_CONTEUDO_RESPOSTA, PROMPT_QUERY_INFORMACOES_RELEVANTES_SOBRE]
    for prompt in prompts_queries:
        new_queries.append(generate_new_query(prompt=prompt, model_id=state.model_id,original_question=search_query))


    all_queries = [search_query] + new_queries 

    all_retrieved_docs = []
    q_vectors = []
    for query in all_queries:
        q_vectors.append(embedder.embed_query(query))
    num_docs_question = k_docs // 2
    num_docs_per_query = (k_docs-num_docs_question) // 4

    for i, q_vector in enumerate(q_vectors):
        if i==0:
            retrieved_docs = vectorstore.similarity_search_with_score_by_vector(
                embedding=q_vector,
                k=num_docs_question,      
                fetch_k=fetch_k,
                kwargs={"score_threshold": threshold}
            )
            all_retrieved_docs.extend(retrieved_docs)

            q_vector = []
        else:
            retrieved_docs = vectorstore.similarity_search_with_score_by_vector(
                embedding=q_vector,
                k=num_docs_per_query,
                fetch_k=fetch_k,
                kwargs={"score_threshold": threshold}
            )
            all_retrieved_docs.extend(retrieved_docs)

            q_vector = []


    unique_docs = {}
    for doc in all_retrieved_docs:
        unique_docs[doc[0].page_content] = doc

    docs = list(unique_docs.values())
    proc_docs = []
    for doc in docs:
        titulo = doc[0].metadata.get("titulo", "Sem título")
        arquivo = doc[0].metadata.get("arquivo_origem", "")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc[0].page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"url": url, "title": titulo}))
    
    new_history = state.chat_history
    if len(new_history)> 2:
        if isinstance(new_history[len(new_history)-1], AIMessage) and new_history[len(new_history)-2].content != search_query:
            new_history.append(HumanMessage(search_query))
        else:  
            if search_query!=new_history[len(new_history)-1].content:
                new_history.append(HumanMessage(search_query))
    else:
        new_history.append(HumanMessage(search_query))
    return {"documents": proc_docs, "chat_history": new_history, "search_method": "similarity_threshold"}


def retrieve_docs_mmr_node(state: GraphState, vectorstore: FAISS, EMBEDDING_MODEL: str, k_docs: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    print("Retrieving...\n")
    search_query = state.question

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device':'cpu'})

    q_vector = embedder.embed_query(search_query)
    new_queries = []
    prompts_queries = [PROMPT_QUERY_ASSUNTO,PROMPT_QUERY_OBJETIVO, PROMPT_QUERY_CONTEUDO_RESPOSTA, PROMPT_QUERY_INFORMACOES_RELEVANTES_SOBRE]
    for prompt in prompts_queries:
        new_queries.append(generate_new_query(prompt=prompt, model_id=state.model_id,original_question=search_query))


    all_queries = [search_query] + new_queries 

    all_retrieved_docs = []
    q_vectors = []
    for query in all_queries:
        q_vectors.append(embedder.embed_query(query))
    
    num_docs_question = k_docs // 2
    num_docs_per_query = (k_docs - num_docs_question) // 4

    for i, q_vector in enumerate(q_vectors):
        if i==0:
            retrieved_docs = vectorstore.max_marginal_relevance_search_by_vector(
                embedding=q_vector,
                k=num_docs_question,  
                lambda_mult=lambda_mult,    
                fetch_k=fetch_k
            )
            all_retrieved_docs.extend(retrieved_docs)

            q_vector = []
        else:
            retrieved_docs = vectorstore.max_marginal_relevance_search_by_vector(
                embedding=q_vector,
                k=num_docs_per_query,
                lambda_mult=lambda_mult,
                fetch_k=fetch_k
            )
            all_retrieved_docs.extend(retrieved_docs)

            q_vector = []


    unique_docs = {}
    for doc in all_retrieved_docs:
        unique_docs[doc[0].page_content] = doc[0]

    docs = list(unique_docs.values())
    
    proc_docs = []
    for doc in docs:
        titulo = doc[0].metadata.get("titulo", "Sem título")
        arquivo = doc[0].metadata.get("arquivo_origem", "")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc[0].page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"url": url, "title": titulo}))
    
    new_history = state.chat_history
    if len(new_history)> 2:
        if isinstance(new_history[len(new_history)-1], AIMessage) and new_history[len(new_history)-2].content != search_query:
            new_history.append(HumanMessage(search_query))
        else:  
            if search_query!=new_history[len(new_history)-1].content:
                new_history.append(HumanMessage(search_query))
    else:
        new_history.append(HumanMessage(search_query))
    return {"documents": proc_docs, "chat_history": new_history, "search_method": "mmr"}

def reset_answer(state: GraphState):
    return {"answer": "", "documents": [], "sources": [],"search_method": "similarity_threshold", "gen_lock": False}

def resume_historico(state: GraphState):
    print("Resuming...\n")
    history = []
    for i in state.chat_history:
        if isinstance(i,HumanMessage):
            history.append(f"Humano: '{i.content}'")
        elif isinstance(i,AIMessage):
            history.append(f"IA: '{i.content}'")
    formatted_hist = "\n".join(history)

    model_id = state.model_id
    chat = get_model_from_cache(model_id)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_RESUMO_HISTORICO)
    chain = prompt_template.pipe(chat).pipe(StrOutputParser())
    with torch.no_grad():
        resp = chain.invoke({
            "historico": formatted_hist
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    
    if state.model_id=="mygemma3:12b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision:11b":
        fResp = _internal_extract_resposta_llama(resp)
    new_history = []
    new_history.append(AIMessage(fResp))
    return {"chat_history": new_history}

def verify_history(state: GraphState):
    if len(state.chat_history) >= 1:
        return "yes"
    else:
        return "no"
    
def verify_context(state:GraphState):
    context = state.chat_history
    pergunta = state.question
    model_id = state.model_id
    chat = get_model_from_cache(model_id)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_AVALIAR_CONTEXTO)
    chain = prompt_template.pipe(chat).pipe(StrOutputParser())
    with torch.no_grad():
        resp = chain.invoke({
            "contexto": context,
            "pergunta": pergunta
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    
    if state.model_id=="mygemma3:12b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision:11b":
        fResp = _internal_extract_resposta_llama(resp)
    if "não" in fResp: 
        return {"chat_history": []}
    else:
        return {"chat_history": context}
    

def generate_answer_without_history(state:GraphState):
    print("Generating...\n")
    model_id = state.model_id
    chat = get_model_from_cache(model_id)

    question = state.question
    docs = state.documents
    
    all_content = "\n\n---\n\n".join([d.page_content for d in docs])
    

    

    prompt_template = ChatPromptTemplate.from_template(PROMPT_ANSWER_WITHOUT_HISTORY)
    
    chain = prompt_template.pipe(chat).pipe(StrOutputParser())
    with torch.no_grad():
        resp = chain.invoke({
            "pergunta": question, 
            "contexto": all_content
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if state.model_id=="mygemma3:12b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision:11b":
        fResp = _internal_extract_resposta_llama(resp)
    new_history = state.chat_history
    if isinstance(new_history[len(new_history)-1], AIMessage):
        new_history.pop(len(new_history)-1)
        new_history.append(AIMessage(fResp))
    else:
        new_history.append(AIMessage(fResp))
    return {"answer": fResp, "chat_history": new_history}

def verify_gen_lock(state: GraphState):
    if state.gen_lock:
        return "yes"
    else:
        return "no"

def gen_answer_state(state:GraphState):
    return

def gen_prev_hist(state:GraphState):
    return

def generate_choice(state:GraphState):
    if state.gen_lock:
        return "end"
    
    if "no" in decide_to_generate(state):
       if "yes" in verify_history(state):
            return "gen_fault"
       return "gen_wout_hist"
    elif "no_docs_try_mmr" in decide_to_generate(state):
        return "mmr"
    elif "no_docs_try_sim" in decide_to_generate(state):
        return "similarity"
    elif "generate" in decide_to_generate(state):
        if "yes" in verify_history(state):
            return "generate"
        return "gen_wout_hist"
    elif "end" in decide_to_generate(state):
        return "end"
    
def gen_choice_state(state:GraphState):
    return

def verify_lock_state(state:GraphState):
    return

def gen_locker(state:GraphState):
    return {"gen_lock": True}

