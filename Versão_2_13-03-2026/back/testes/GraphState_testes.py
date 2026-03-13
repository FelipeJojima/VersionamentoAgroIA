# Coloque este código no topo do seu arquivo
import psutil
import functools
import os
import torch
import time  # <<< NOVO: Importa a biblioteca de tempo
from pynvml import *

# Funções de ajuda para formatação
def bytes_to_mb(bytes_value):
    return f"{bytes_value / (1024 * 1024):.2f} MB"

def bytes_to_gb(bytes_value):
    return f"{bytes_value / (1024 * 1024 * 1024):.2f} GB"

# DECORADOR FINAL com monitoramento de RAM, PICO de VRAM e TEMPO DE EXECUÇÃO
def monitor_recursos(func):
    """
    Decorador que monitora o uso de RAM, VRAM (incluindo pico)
    e o tempo de execução da função.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        
        process = psutil.Process(os.getpid())
        ram_process_before = process.memory_info().rss
        ram_system_before = psutil.virtual_memory()

        print(f"\n--- [MONITOR] Iniciando '{func.__name__}' ---")
        print(f"    -> RAM do Processo: {bytes_to_mb(ram_process_before)}")
        print(f"    -> RAM Total Sistema: {ram_system_before.percent}% ({bytes_to_gb(ram_system_before.used)} / {bytes_to_gb(ram_system_before.total)})")

        if torch.cuda.is_available():
            try:
                nvmlInit()
                torch_device_count = torch.cuda.device_count()
                for i in range(torch_device_count):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    print(f"    -> VRAM GPU {i} ({nvmlDeviceGetName(handle)}): "
                          f"{(mem_info.used / mem_info.total * 100):.2f}% ({bytes_to_gb(mem_info.used)} / {bytes_to_gb(mem_info.total)})")
                
                try:
                    for i in range(torch_device_count):
                        print(f"    -> VRAM Alocada (PyTorch) GPU {i}: {bytes_to_mb(torch.cuda.memory_allocated(i))}")
                        torch.cuda.reset_peak_memory_stats(i)
                except RuntimeError as e:
                    if "Invalid device argument" in str(e):
                        print("    -> Aviso: Contexto CUDA ainda não inicializado pelo PyTorch. Medição de pico será feita nos próximos passos.")
                    else:
                        raise e
            
            except NVMLError as error:
                print(f"    -> Aviso: Não foi possível monitorar a VRAM via NVML: {error}")
            finally:
                try:
                    nvmlShutdown()
                except NVMLError:
                    pass

        result = func(*args, **kwargs)

        end_time = time.perf_counter()
        duration = end_time - start_time
        
        ram_process_after = process.memory_info().rss
        ram_system_after = psutil.virtual_memory()
        ram_process_diff = ram_process_after - ram_process_before

        print(f"--- [MONITOR] Finalizado '{func.__name__}' ---")
        print(f"    -> Duração da Execução: {duration:.4f} segundos")
        print(f"    -> RAM do Processo: {bytes_to_mb(ram_process_after)} (Variação: +{bytes_to_mb(ram_process_diff)})")
        print(f"    -> RAM Total Sistema: {ram_system_after.percent}% ({bytes_to_gb(ram_system_after.used)} / {bytes_to_gb(ram_system_after.total)})")

        if torch.cuda.is_available():
            try:
                nvmlInit()
                torch_device_count = torch.cuda.device_count()
                for i in range(torch_device_count):
                    handle = nvmlDeviceGetHandleByIndex(i)
                    mem_info = nvmlDeviceGetMemoryInfo(handle)
                    print(f"    -> VRAM GPU {i} ({nvmlDeviceGetName(handle)}): "
                          f"{(mem_info.used / mem_info.total * 100):.2f}% ({bytes_to_gb(mem_info.used)} / {bytes_to_gb(mem_info.total)})")
                    
                    allocated_after = torch.cuda.memory_allocated(i)
                    peak_vram = torch.cuda.max_memory_allocated(i)
                    print(f"    -> VRAM Alocada (Final) GPU {i}: {bytes_to_mb(allocated_after)}")
                    print(f"    -> PICO de VRAM Alocada (PyTorch) GPU {i}: {bytes_to_mb(peak_vram)}")
            except NVMLError as error:
                print(f"    -> Aviso: Não foi possível monitorar a VRAM via NVML: {error}")
            finally:
                try:
                    nvmlShutdown()
                except NVMLError:
                    pass
        print("-" * 60)

        return result
    return wrapper
# --------------------------------------------------------------------------
# SEU CÓDIGO ORIGINAL COM O DECORADOR APLICADO
# --------------------------------------------------------------------------

from pydantic import BaseModel, Field
import torch
from typing import List, Any, Dict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from extract_functions import _internal_extract_resposta_gemma, _internal_extrair_descricao, _internal_extract_resposta_llama
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import MllamaForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, Gemma2ForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from prompts import PROMPT_GRADER, PROMPT_REVIEW, PROMPT_ANSWER, PROMPT_ANSWER_FAULT

# Configuração do PyTorch (mantida do original)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class GraphState(BaseModel):
    question: str
    model_id: str
    answer: str = ""
    search_method: str = "similarity"
    documents: List[Document] = Field(default_factory=list)
    sources: List[Dict[str, str]] = Field(default_factory=list)
    chat_history: List[BaseMessage] = Field(default_factory=list)




_model_cache = {}


def formatar_historico(hist: List[BaseMessage], max_messages: int) -> string:
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




# Aplicando o monitorador nas funções mais pesadas
@monitor_recursos
def get_model_from_cache_gemma(model_id: str):
    """
    Verifica o cache. Se o modelo não estiver lá, ele o carrega,
    armazena no cache e o retorna.
    """
    global _model_cache
    if model_id not in _model_cache:
        bnbConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
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
        
        _model_cache[model_id] = ChatHuggingFace(llm=hf_pipe)
        
    return _model_cache[model_id]

@monitor_recursos
def get_model_from_cache_llama(model_id: str):
    """
    Verifica o cache. Se o modelo não estiver lá, ele o carrega,
    armazena no cache e o retorna.
    """
    global _model_cache
    if model_id not in _model_cache:
        bnbConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        model_llm = MllamaForCausalLM.from_pretrained(
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
        
        _model_cache[model_id] = ChatHuggingFace(llm=hf_pipe)

    return _model_cache[model_id]


def get_model_from_cache(model_id:str):
    if model_id=="google/gemma-2-9b-it":
        return get_model_from_cache_gemma(model_id)
    elif model_id=="meta-llama/Llama-3.2-11B-Vision-Instruct":
        return get_model_from_cache_llama(model_id)
    
@monitor_recursos
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

@monitor_recursos
def retrieve_docs_similarity_node(state: GraphState,vectorstore: FAISS, k_docs: int = 4, fetch_k: int = 30) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    # print("\n\n\nSIM\n\n")
    question = state.question
    chat_history = state.chat_history

    if chat_history:
        history_text = formatar_historico(chat_history, 6)
        search_query = f"Contexto da Conversa:\n{history_text}\n\nPergunta Atual: {question}"
    else:
        search_query = question


    
    retriever = vectorstore.as_retriever(search_type="similarity",search_kwargs={"k": k_docs, "fetch_k": fetch_k})
    docs = retriever.invoke(search_query)
    
    proc_docs = []
    for doc in docs:
        # print(doc)
        titulo = doc.metadata.get("titulo", "Sem título")
        arquivo = doc.metadata.get("arquivo_origem", "")
        # print(f"Arquivo {arquivo}. \n")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc.page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"url": url, "title": titulo}))
    new_history = state.chat_history
    if len(new_history)> 2:
        if isinstance(new_history[len(new_history)-1], AIMessage) and new_history[len(new_history)-2].content != question:
            new_history.append(HumanMessage(question))
        else:  
            if question!=new_history[len(new_history)-1].content:
                new_history.append(HumanMessage(question))
    else:
        new_history.append(HumanMessage(question))
    return {"documents": proc_docs, "chat_history": new_history, "search_method": "similarity"}

@monitor_recursos
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
        with torch.no_grad():
            response = chain.invoke({"doc": d.page_content, "pergunta": question})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if state.model_id=="google/gemma-2-9b-it":
            lResponse = _internal_extract_resposta_gemma(response.lower())
        elif state.model_id=="meta-llama/Llama-3.2-11B-Vision-Instruct":
            lResponse = _internal_extract_resposta_llama(response.lower())

        if "sim" in lResponse:
            rel_docs.append(d)
            sources.append({"url": d.metadata.get("url", ""), "title": d.metadata.get("title", "Fonte desconhecida")})
    
    return {"documents": rel_docs, "sources": sources}

@monitor_recursos
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
    if state.model_id=="google/gemma-2-9b-it":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="meta-llama/Llama-3.2-11B-Vision-Instruct":
        fResp = _internal_extract_resposta_llama(resp)
    new_history = state.chat_history
    if isinstance(new_history[len(new_history)-1], AIMessage):
        new_history.pop(len(new_history)-1)
        new_history.append(AIMessage(fResp))
    else:
        new_history.append(AIMessage(fResp))
    return {"answer": fResp, "chat_history": new_history}

@monitor_recursos
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
    with torch.no_grad():
        response = chain.invoke({"pergunta": question, "resposta": answer})
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if state.model_id=="google/gemma-2-9b-it":
        finalResponse = _internal_extract_resposta_gemma(response.lower())
    elif state.model_id=="meta-llama/Llama-3.2-11B-Vision-Instruct":
        finalResponse = _internal_extract_resposta_llama(response.lower())

    if "não" in finalResponse and "A resposta anterior foi considerada insatisfatória, refaça." not in response:
        if state.search_method == "mmr":
            if state.answer=="":
                return "não"
            else:
                return "end"
        if state.search_method == "similarity_threshold":
            return "MMR"
        return "Sim_Threshold"
    else:
        return "sim"


def decide_to_generate(state: GraphState) -> str:
    """
    Decide o próximo passo: gerar com documentos, gerar apenas com histórico, ou terminar.
    """
    if state.documents:
        return "generate"
    else:
        if state.search_method=="similarity":
            return "no_docs_try_threshold"
        elif state.search_method=="similarity_threshold":
            return "no_docs_try_mmr"
        else:
            if state.answer=="":
                return "no"
            else:
                return "end"

@monitor_recursos
def gen_answer_fault(state: GraphState):
    model_id = state.model_id
    chat = get_model_from_cache(model_id)

    question = state.question

    prompt_template = ChatPromptTemplate.from_template(PROMPT_ANSWER_FAULT)
    
    chain = prompt_template.pipe(chat).pipe(StrOutputParser())
    
    resp = chain.invoke({
        "pergunta": question, 
    })
    
    if state.model_id=="google/gemma-2-9b-it":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="meta-llama/Llama-3.2-11B-Vision-Instruct":
        fResp = _internal_extract_resposta_llama(resp)
    new_history = state.chat_history
    new_history.append(AIMessage(fResp))
    return {"answer": fResp, "chat_history": new_history}


@monitor_recursos
def retrieve_docs_similarity_threshold_node(state: GraphState, vectorstore: FAISS, threshold: float = 0.5,  k_docs: int = 4,fetch_k: int = 30) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    question = state.question
    chat_history = state.chat_history
    # print("\n\n\nTHRESHOLD\n\n")
    if chat_history:
        history_text = formatar_historico(chat_history, 6)
        search_query = f"Contexto da Conversa:\n{history_text}\n\nPergunta Atual: {question}"
    else:
        search_query = question
    

    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": k_docs, "score_threshold": threshold, "fetch_k":fetch_k})
    docs = retriever.invoke(search_query)
    
    proc_docs = []
    for doc in docs:
        # print(doc)
        titulo = doc.metadata.get("titulo", "Sem título")
        arquivo = doc.metadata.get("arquivo_origem", "")
        # print(f"Arquivo {arquivo}. \n")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc.page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"source": url}))
    
    new_history = state.chat_history
    if len(new_history)> 2:
        if isinstance(new_history[len(new_history)-1], AIMessage) and new_history[len(new_history)-2].content != question:
            new_history.append(HumanMessage(question))
        else:  
            if question!=new_history[len(new_history)-1].content:
                new_history.append(HumanMessage(question))
    else:
        new_history.append(HumanMessage(question))
    return {"documents": proc_docs, "chat_history": new_history, "search_method": "similarity_threshold"}


@monitor_recursos
def retrieve_docs_mmr_node(state: GraphState, vectorstore: FAISS, k_docs: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    question = state.question
    chat_history = state.chat_history
    # print("\n\n\nMMR\n\n")
    
    if chat_history:
        history_text = formatar_historico(chat_history, 6)
        search_query = f"Contexto da Conversa:\n{history_text}\n\nPergunta Atual: {question}"
    else:
        search_query = question
    
    retriever = vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": k_docs, "fetch_k": fetch_k, "lambda_mult":lambda_mult})
    docs = retriever.invoke(search_query)
    
    proc_docs = []
    for doc in docs:
        # print(doc)
        titulo = doc.metadata.get("titulo", "Sem título")
        arquivo = doc.metadata.get("arquivo_origem", "")
        # print(f"Arq {arquivo}\n")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc.page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"source": url}))
    
    new_history = state.chat_history
    if len(new_history)> 2:
        if isinstance(new_history[len(new_history)-1], AIMessage) and new_history[len(new_history)-2].content != question:
            new_history.append(HumanMessage(question))
        else:  
            if question!=new_history[len(new_history)-1].content:
                new_history.append(HumanMessage(question))
    else:
        new_history.append(HumanMessage(question))
    return {"documents": proc_docs, "chat_history": new_history, "search_method": "mmr"}

@monitor_recursos
def reset_answer(state: GraphState):
    return {"answer": "", "documents": [], "sources": [],"search_method": "similarity"}

