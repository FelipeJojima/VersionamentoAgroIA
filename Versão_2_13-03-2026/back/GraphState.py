from pydantic import BaseModel, Field
import torch
from typing import List, Any, Dict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from .extract_functions import _internal_extract_resposta, _internal_extrair_descricao
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import Gemma2ForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from .prompts import PROMPT_GRADER, PROMPT_REVIEW, PROMPT_ANSWER, PROMPT_ANSWER_W_HISTORY
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class GraphState(BaseModel):
    question: str
    model_id: str
    answer: str = ""
    documents: List[Document] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    chat_history: List[BaseMessage] = Field(default_factory=list)

# O cache agora é um dicionário para armazenar múltiplos modelos pelo seu ID
_model_cache = {}

def get_model_from_cache(model_id: str):
    """
    Verifica o cache. Se o modelo não estiver lá, ele o carrega,
    armazena no cache e o retorna.
    """
    global _model_cache
    if model_id not in _model_cache:
        #print(f"\n--- MODELO '{model_id}' NÃO ENCONTRADO NO CACHE. CARREGANDO... ---")
        bnbConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_llm = Gemma2ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, quantization_config=bnbConfig, device_map="auto", low_cpu_mem_usage=True, use_safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    
        pipe = pipeline(
            "text-generation",
            model=model_llm,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.25,
            do_sample=True,
            repetition_penalty=1.2,
            penalty_alpha= 0.7,
            top_k= 10,
        )
        hf_pipe = HuggingFacePipeline(pipeline=pipe)
        
        # Armazena a nova instância do modelo no cache
        _model_cache[model_id] = ChatHuggingFace(llm=hf_pipe)
        #print(f"--- MODELO '{model_id}' CARREGADO E ARMAZENADO NO CACHE. ---")
    #else:
        #print(f"\n--- Usando modelo '{model_id}' do cache. ---")
        
    return _model_cache[model_id]
   
def create_model_node(state: GraphState) -> None:
    """
    Este nó age como um "pré-aquecedor". Ele garante que o modelo
    especificado no estado (`model_id`) esteja carregado no cache.
    """
    #print("\n--- NÓ: create_model (Cache Warmer) ---")
    model_id = state.model_id
    if not model_id:
        raise ValueError("model_id não foi fornecido no estado inicial do grafo.")
    
    get_model_from_cache(model_id)
    return

def retrieve_docs_similarity_node(state: GraphState, embedding: HuggingFaceEmbeddings, faiss_path: str, k_docs: int = 4) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    question = state.question
    chat_history = state.chat_history
    print(chat_history)
    if chat_history:
        history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        print(history_text)
        search_query = f"Contexto da Conversa:\n{history_text}\n\nPergunta Atual: {question}"
    else:
        search_query = question
    
    try:
        vectorstore = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o índice FAISS de '{faiss_path}'. {e}")
        return {"documents": []}
    
    # Substituído similarity_search por as_retriever().invoke()
    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": k_docs})
    docs = retriever.invoke(search_query)
    
    proc_docs = []
    for doc in docs:
        titulo = doc.metadata.get("titulo", "Sem título")
        arquivo = doc.metadata.get("arquivo_origem", "")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc.page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"url": url, "title": titulo}))
    new_history = state.chat_history
    new_history.append(HumanMessage(question))
    return {"documents": proc_docs, "chat_history": new_history}

def grading_node(state: GraphState) -> Dict[str, Any]:
    """
    Nó que avalia os documentos recuperados do FAISS. Retorna os documentos que o modelo avaliar relevante para a pergunta do usuário.
    """

    model_id = state.model_id
    chat = get_model_from_cache(model_id)
    
    question = state.question
    docs = state.documents

    if not docs:
        return {"documents": [], "sources": []}

    rel_docs = []
    sources = []
    prompt = ChatPromptTemplate.from_template(PROMPT_GRADER)
    chain = prompt.pipe(chat).pipe(StrOutputParser())
    
    for d in docs:
        response = chain.invoke({"doc": d.page_content, "pergunta": question})
        lResponse = _internal_extract_resposta(response.lower())
        print(lResponse)
        if "sim" in lResponse:
            rel_docs.append(d)
            sources.append({"url": d.metadata.get("url", ""), "title": d.metadata.get("title", "Fonte desconhecida")})
    
    return {"documents": rel_docs, "sources": sources}

def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """
    Nó que gera a resposta para a pergunta a partir dos documentos e do histórico do chat.
    """
    model_id = state.model_id
    chat = get_model_from_cache(model_id)

    question = state.question
    docs = state.documents
    chat_history = state.chat_history
    
    all_content = "\n\n---\n\n".join([d.page_content for d in docs])
    
        # --- LÓGICA DE FORMATAÇÃO ADICIONADA AQUI ---
    def format_history_for_prompt(history: list) -> str:
        if not history:
            return "Nenhum histórico de conversa ainda."
        
        formatted_lines = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                formatted_lines.append(f"Usuário: {msg.content}")
            elif isinstance(msg, AIMessage):
                # A MUDANÇA CRÍTICA: Re-rotulamos a mensagem da IA
                formatted_lines.append(f"Sua Resposta Anterior: {msg.content}")
        return "\n".join(formatted_lines)
    
    formatted_history = format_history_for_prompt(chat_history)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_ANSWER)
    
    chain = prompt_template.pipe(chat).pipe(StrOutputParser())
    
    resp = chain.invoke({
        "pergunta": question, 
        "contexto": all_content,
        "historico": formatted_history
    })
    
    fResp = _internal_extract_resposta(resp)
    new_history = state.chat_history
    new_history.append(AIMessage(fResp))
    return {"answer": fResp, "chat_history": new_history}

def review_answer_node(state: GraphState) -> str:
    """
    Nó que avalia a resposta, verificando se a questão do usuário foi respondida.
    """
    model_id = state.model_id
    chat = get_model_from_cache(model_id)

    question = state.question
    answer = state.answer
    
    prompt = ChatPromptTemplate.from_template(PROMPT_REVIEW)
    chain = prompt.pipe(chat).pipe(StrOutputParser())
    
    response = chain.invoke({"pergunta": question, "resposta": answer})
    finalResponse = _internal_extract_resposta(response.lower())

    if "não" in finalResponse and "A resposta anterior foi considerada insatisfatória, refaça." not in response:
        if "MMR" in finalResponse:
            return "não"
        if "Similarity_Threshold" in finalResponse:
            return "MMR"
        return "Sim_Threshold"
    else:
        return "sim"

def decide_to_generate(state: GraphState) -> str:
    """
    Decide o próximo passo: gerar com documentos, gerar apenas com histórico, ou terminar.
    """
    print("\n--- DECISÃO: Gerar, Usar Histórico ou Terminar? ---")

    if state.documents:
        # 1. Se há documentos, use-os para gerar a resposta.
        print("Resultado: Documentos encontrados. Prosseguindo para 'generate'.")
        return "generate"
    elif state.chat_history:
        # 2. Se NÃO há documentos, MAS HÁ histórico, use apenas o histórico.
        print("Resultado: Sem documentos, mas com histórico. Prosseguindo para 'gen_only_w_chat_history'.")
        return "gen_only_w_chat_history"
    else:
        # 3. Se não há nem documentos nem histórico, não há como responder.
        print("Resultado: Sem documentos e sem histórico. Impossível gerar resposta.")
        return "end"
    
def modify_question_for_retry_to_mmr_node(state: GraphState) -> Dict[str, str]:
    """
    Nó que modifica a pergunta para incluir a instrução de refazer,
    garantindo que a instrução não seja duplicada.
    """
    #print("\n--- NÓ: modify_question_for_retry ---")
    
    current_question = state.question
    instrucao_refazer = f"\nUtilize o MMR.\n"
    
    if instrucao_refazer not in current_question:
        new_question = current_question + instrucao_refazer
    else:
        new_question = current_question # Não faz nada se a instrução já estiver lá

    return {"question": new_question}

def modify_question_final_node(state: GraphState) -> Dict[str, str]:
    """
    Nó que modifica a pergunta para incluir a instrução de refazer,
    garantindo que a instrução não seja duplicada.
    """
    #print("\n--- NÓ: modify_question_for_retry ---")
    
    current_question = state.question
    instrucao_refazer = f"\nA resposta anterior foi considerada insatisfatória, refaça.\n<resposta_anterior>{state.answer}</resposta_anterior>\n"
    
    if instrucao_refazer not in current_question:
        new_question = current_question + instrucao_refazer
    else:
        new_question = current_question # Não faz nada se a instrução já estiver lá

    return {"question": new_question}

def modify_question_for_retry_to_threshold_node(state: GraphState) -> Dict[str, str]:
    """
    Nó que modifica a pergunta para incluir a instrução de refazer,
    garantindo que a instrução não seja duplicada.
    """
    #print("\n--- NÓ: modify_question_for_retry ---")
    
    current_question = state.question
    instrucao_refazer = f"\nUtilize o Similarity_Threshold.\n"
    
    if instrucao_refazer not in current_question:
        new_question = current_question + instrucao_refazer
    else:
        new_question = current_question # Não faz nada se a instrução já estiver lá

    return {"question": new_question}

def generate_answer_w_history_node(state: GraphState) -> Dict[str, Any]:
    """
    Nó que gera a resposta para a pergunta a partir do histórico do chat.
    """
    model_id = state.model_id
    chat = get_model_from_cache(model_id)

    question = state.question
    chat_history = state.chat_history
    

        # --- LÓGICA DE FORMATAÇÃO ADICIONADA AQUI ---
    def format_history_for_prompt(history: list) -> str:
        if not history:
            return "Nenhum histórico de conversa ainda."
        
        formatted_lines = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                formatted_lines.append(f"Usuário: {msg.content}")
            elif isinstance(msg, AIMessage):
                # A MUDANÇA CRÍTICA: Re-rotulamos a mensagem da IA
                formatted_lines.append(f"Sua Resposta Anterior: {msg.content}")
        return "\n".join(formatted_lines)
    
    formatted_history = format_history_for_prompt(chat_history)


    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_ANSWER_W_HISTORY)
    
    chain = prompt_template.pipe(chat).pipe(StrOutputParser())
    
    resp = chain.invoke({
        "pergunta": question, 
        "historico": formatted_history
    })
    
    fResp = _internal_extract_resposta(resp)
    new_history = state.chat_history
    new_history.append(AIMessage(fResp))
    return {"answer": fResp, "chat_history": new_history}

def retrieve_docs_similarity_threshold_node(state: GraphState, embedding: HuggingFaceEmbeddings, faiss_path: str,threshold: float = 0.5,  k_docs: int = 4) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    question = state.question
    chat_history = state.chat_history
    
    if chat_history:
        history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        print(history_text)
        search_query = f"Contexto da Conversa:\n{history_text}\n\nPergunta Atual: {question}"
    else:
        search_query = question
    
    try:
        vectorstore = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o índice FAISS de '{faiss_path}'. {e}")
        return {"documents": []}
    
    # Substituído similarity_search por as_retriever().invoke()
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": k_docs, "score_threshold": threshold})
    docs = retriever.invoke(search_query)
    
    proc_docs = []
    for doc in docs:
        titulo = doc.metadata.get("titulo", "Sem título")
        arquivo = doc.metadata.get("arquivo_origem", "")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc.page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"source": url}))
    
    new_history = state.chat_history
    new_history.append(HumanMessage(question))
    return {"documents": proc_docs, "chat_history": new_history}

def retrieve_docs_mmr_node(state: GraphState, embedding: HuggingFaceEmbeddings, faiss_path: str, k_docs: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    question = state.question
    chat_history = state.chat_history
    
    if chat_history:
        history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        print(history_text)
        search_query = f"Contexto da Conversa:\n{history_text}\n\nPergunta Atual: {question}"
    else:
        search_query = question
    
    try:
        vectorstore = FAISS.load_local(faiss_path, embeddings=embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o índice FAISS de '{faiss_path}'. {e}")
        return {"documents": []}
    
    # Substituído similarity_search por as_retriever().invoke()

    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": k_docs, "fetch_k": fetch_k, "lambda_mult":lambda_mult})
    docs = retriever.invoke(search_query)
    
    proc_docs = []
    for doc in docs:
        titulo = doc.metadata.get("titulo", "Sem título")
        arquivo = doc.metadata.get("arquivo_origem", "")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc.page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"source": url}))
    
    new_history = state.chat_history
    new_history.append(HumanMessage(question))
    return {"documents": proc_docs, "chat_history": new_history}