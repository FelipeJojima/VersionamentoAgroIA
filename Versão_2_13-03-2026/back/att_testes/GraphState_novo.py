# Codigo para resgate de informações
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
# graphstate
# --------------------------------------------------------------------------

from pydantic import BaseModel, Field
import torch
import asyncio
from typing import List, Any, Dict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from extract_functions import _internal_extract_resposta_gemma, _internal_extrair_descricao, _internal_extract_resposta_llama
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# from transformers import MllamaForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig, Gemma2ForCausalLM
# from langchain_community.llms import VLLM
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from prompts import PROMPT_GRADER, PROMPT_REVIEW, PROMPT_ANSWER, PROMPT_ANSWER_FAULT, PROMPT_RESUMO_HISTORICO, PROMPT_AVALIAR_CONTEXTO, PROMPT_ANSWER_WITHOUT_HISTORY, PROMPT_QUERY_ASSUNTO,PROMPT_QUERY_OBJETIVO,PROMPT_QUERY_PONTOS_PRINCIPAIS, PROMPT_QUERY_CONTEUDO_RESPOSTA, PROMPT_QUERY_INFORMACOES_RELEVANTES_SOBRE
from langchain_community.tools import DuckDuckGoSearchResults
from ragas import EvaluationDataset, evaluate, RunConfig
# from ragas.llms.base import instructor_llm_factory
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy, Faithfulness, ResponseGroundedness, AspectCritic
from ragas.metrics import ContextRelevance #metrica de avaliacao de documentos
from ragas.metrics import SummarizationScore #metrica que avalia o resumo
# from ragas.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings
from ragas.embeddings.base import LangchainEmbeddingsWrapper
# Configuração do PyTorch (mantida do original)
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
# @monitor_recursos
def get_model_from_cache_gemma(model_id: str):
    """
    Verifica o cache. Se o modelo não estiver lá, ele o carrega,
    armazena no cache e o retorna.
    """
    global _model_cache
    if model_id not in _model_cache:
        # bnbConfig = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True
        # )
        # model_llm = Gemma2ForCausalLM.from_pretrained(
        #     model_id, torch_dtype=torch.bfloat16, quantization_config=bnbConfig, device_map="auto", low_cpu_mem_usage=True, use_safetensors=True
        # )
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
    
        # pipe = pipeline(
        #     "text-generation",
        #     model=model_llm,
        #     tokenizer=tokenizer,
        #     max_new_tokens=2048,
        #     temperature=0.25,
        #     do_sample=True,
        #     repetition_penalty=1.2,
        #     penalty_alpha= 0.7,
        #     top_k= 10,
        # )

        model = OllamaLLM(model=model_id, temperature=0.25,top_k=10,top_p=0.7,repeat_penalty=1.2)
        # hf_pipe = HuggingFacePipeline(pipeline=pipe)
        
        # _model_cache[model_id] = ChatHuggingFace(llm=hf_pipe)
        _model_cache[model_id] = model
        
    return _model_cache[model_id]

# @monitor_recursos
def get_model_from_cache_llama(model_id: str):
    """
    Verifica o cache. Se o modelo não estiver lá, ele o carrega,
    armazena no cache e o retorna.
    """
    global _model_cache
    if model_id not in _model_cache:
    #     bnbConfig = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16,
    #         bnb_4bit_use_double_quant=True
    #     )
    #     model_llm = MllamaForCausalLM.from_pretrained(
    #         model_id, torch_dtype=torch.bfloat16, quantization_config=bnbConfig, device_map="auto", low_cpu_mem_usage=True, use_safetensors=True
    #     )
    #     tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    #     pipe = pipeline(
    #         "text-generation",
    #         model=model_llm,
    #         tokenizer=tokenizer,
    #         max_new_tokens=2048,
    #         temperature=0.25,
    #         do_sample=True,
    #         repetition_penalty=1.2,
    #         penalty_alpha= 0.7,
    #         top_k= 10,
    #     )
    #     hf_pipe = HuggingFacePipeline(pipeline=pipe)
        
    #     _model_cache[model_id] = ChatHuggingFace(llm=hf_pipe)
        # model_llm = VLLM(
        #     model=model_id,
        #     trust_remote_code= True,
        #     max_new_tokens=2048,
        #     temperature=0.25,
        #     top_k=10,
        #     frequence_penalty=1.2,
        #     presence_penalty=0.7,
        #     dtype=torch.bfloat16,
        #     vllm_kwargs={"quantization":"bitsandbytes"}
        # )
        model = OllamaLLM(model=model_id, temperature=0.25,top_k=10,top_p=0.7,repeat_penalty=1.2)
        _model_cache[model_id] = model

    return _model_cache[model_id]


def get_model_from_cache(model_id:str):
    if model_id=="gemma2:9b":
        return get_model_from_cache_gemma(model_id)
    elif model_id=="myllama32-vision-11b":
        return get_model_from_cache_llama(model_id)
    
# @monitor_recursos
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
        print(resp)
        print("\n\n\n")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if model_id=="gemma2:9b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif model_id=="myllama32-vision-11b":
        fResp = _internal_extract_resposta_llama(resp)

    return fResp


# @monitor_recursos
def retrieve_docs_similarity_node(state: GraphState,vectorstore: FAISS,  EMBEDDING_MODEL: str, k_docs: int = 5, fetch_k: int = 30) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método similarity_search_by_vector() para a busca.
    """
    # print("\n\n\nSIM\n\n")
    search_query = state.question

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device':'cpu'})
    q_vector = embedder.embed_query(search_query)
    new_queries = []
    prompts_queries = [PROMPT_QUERY_ASSUNTO,PROMPT_QUERY_OBJETIVO,PROMPT_QUERY_PONTOS_PRINCIPAIS, PROMPT_QUERY_CONTEUDO_RESPOSTA, PROMPT_QUERY_INFORMACOES_RELEVANTES_SOBRE]
    for prompt in prompts_queries:
        new_queries.append(generate_new_query(prompt=prompt, model_id=state.model_id,original_question=search_query))


    all_queries = [search_query] + new_queries 

    all_retrieved_docs = []

    q_vectors = []
    for query in all_queries:
        print("\n\nEmbedding...\n\n")
        q_vectors.append(embedder.embed_query(query))
    
    num_docs_per_query = k_docs // 5
    print("\n\nStarting retrieval...\n\n")
    for q_vector in q_vectors:
    
        retrieved_docs = vectorstore.similarity_search_by_vector(
            embedding=q_vector,
            k=num_docs_per_query,      
            fetch_k=fetch_k
        )
        all_retrieved_docs.extend(retrieved_docs)

        q_vector = []


    unique_docs = {}
    for doc in all_retrieved_docs:
        unique_docs[doc[0].page_content] = doc

    docs = list(unique_docs.values())
    print("Finished retrieval.\n\n")
    proc_docs = []
    for doc in docs:
        # print(doc)
        titulo = doc[0].metadata.get("titulo", "Sem título")
        arquivo = doc[0].metadata.get("arquivo_origem", "")
        # print(f"Arquivo {arquivo}. \n")
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

# @monitor_recursos
def grading_node(state: GraphState, EMBEDDING_MODEL:str) -> Dict[str, Any]:
    """
    Nó que avalia os documentos recuperados do FAISS. Retorna os documentos que o modelo avaliar relevante para a pergunta do usuário.
    """
    model_id = state.model_id
    chat = get_model_from_cache(model_id)
    print("\n\n\nGRADING...\n\n\n")
    question = state.question
    docs = state.documents

    if not docs:
        return {"documents": [], "sources": []}

    rel_docs = []
    sources = []

    embedder_hf = HuggingFaceEmbeddings(model=EMBEDDING_MODEL,model_kwargs={ 'device':'cpu'})
    embedder = LangchainEmbeddingsWrapper(embeddings=embedder_hf)
    data = []

    

    llm_ragas = get_model_from_cache_gemma(model_id=model_id)
    
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
            # print(results)
        except Exception as e:
            print(f"    -> ERRO RAGAs: Falha ao executar 'evaluate': {e}")
            results = {}
        # print(results['nv_context_relevance'][0])
        data.clear()
        if((results['nv_context_relevance'][0])>=0.5):
            rel_docs.append(d)
    print("Finsihed grading.\n\n")

    
    # prompt = ChatPromptTemplate.from_template(PROMPT_GRADER)
    # chain = prompt.pipe(chat).pipe(StrOutputParser())
    
    # for d in docs:
    #     with torch.no_grad():
    #         response = chain.invoke({"doc": d.page_content, "pergunta": question})
    #     if state.model_id=="google/gemma-2-9b-it":
    #         lResponse = _internal_extract_resposta_gemma(response.lower())
    #     elif state.model_id=="meta-llama/Llama-3.2-11B-Vision-Instruct":
    #         lResponse = _internal_extract_resposta_llama(response.lower())
    #     # print(d)
    #     # print(lResponse)
    #     if "sim" in lResponse:
    #         rel_docs.append(d)
    #         sources.append({"url": d.metadata.get("source", ""), "title": d.metadata.get("title", "Fonte desconhecida")})
    #     # print(sources)
    #     if torch.cuda.is_available():
    #         torch.cuda.empty_cache()
    return {"documents": rel_docs, "sources": sources}

# @monitor_recursos
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
    if state.model_id=="gemma2:9b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision-11b":
        fResp = _internal_extract_resposta_llama(resp)
    new_history = state.chat_history
    if isinstance(new_history[len(new_history)-1], AIMessage):
        new_history.pop(len(new_history)-1)
        new_history.append(AIMessage(fResp))
    else:
        new_history.append(AIMessage(fResp))
    return {"answer": fResp, "chat_history": new_history}

# @monitor_recursos
def review_answer_node(state: GraphState, EMBEDDING_MODEL: str) -> str:
    """
    Nó que avalia a resposta, verificando se a questão do usuário foi respondida.
    """

    ###Versão antiga: LLM-AS-A-JUDGE PURO
    # model_id = state.model_id
    # chat = get_model_from_cache(model_id)

    # question = state.question
    # answer = state.answer
    
    # prompt = ChatPromptTemplate.from_template(PROMPT_REVIEW)
    # chain = prompt.pipe(chat).pipe(StrOutputParser())
    # with torch.no_grad():
    #     response = chain.invoke({"pergunta": question, "resposta": answer})
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    # if state.model_id=="google/gemma-2-9b-it":
    #     finalResponse = _internal_extract_resposta_gemma(response.lower())
    # elif state.model_id=="meta-llama/Llama-3.2-11B-Vision-Instruct":
    #     finalResponse = _internal_extract_resposta_llama(response.lower())

    # if "não" in finalResponse and "A resposta anterior foi considerada insatisfatória, refaça." not in response:
    #     if state.search_method == "mmr":
    #         if state.answer=="":
    #             return "não"
    #         else:
    #             return "end"
    #     if state.search_method == "Similarity":
    #         return "MMR"
    #     return "Similarity"
    # else:
    #     return "sim"


    ### Nova versão com o RAGAs
    query = state.question
    response = state.answer
    documents = state.documents
    embedder_hf = HuggingFaceEmbeddings(model=EMBEDDING_MODEL,model_kwargs={ 'device':'cpu'})
    embedder = LangchainEmbeddingsWrapper(embeddings=embedder_hf)
    # embedder = OllamaEmbeddings(model="gemma2")
    data = []
    data.append(
        {"user_input": query,
         "retrieved_contexts": [doc.page_content for doc in documents],
         "response": response}
    )
    eval_data = EvaluationDataset.from_list(data)

    llm_ragas = get_model_from_cache_gemma(model_id=state.model_id)
    
    evaluator_llm = LangchainLLMWrapper(langchain_llm=llm_ragas)
    runconfig = RunConfig(max_workers=1)

    try:
        results = evaluate(
            dataset=eval_data, 
            metrics=[
                LLMContextPrecisionWithoutReference(llm=evaluator_llm), 
                # ResponseRelevancy(llm=evaluator_llm, embeddings=embedder), 
                # Faithfulness(llm=evaluator_llm), 
                ResponseGroundedness(llm=evaluator_llm), 
                # AspectCritic(name="reviewer", definition="Is the answer useful to solve the user question?", llm=evaluator_llm)
            ],
            llm=evaluator_llm, 
            embeddings=embedder, 
            run_config=runconfig,
            show_progress=False
        )
        # print(results)
    except Exception as e:
        print(f"    -> ERRO RAGAs: Falha ao executar 'evaluate': {e}")
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

    print(f"    -> Score de 'llm_context_precision_without_reference': {score}")    
    
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

# @monitor_recursos
def gen_answer_fault(state: GraphState):
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
    if state.model_id=="gemma2:9b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision-11b":
        fResp = _internal_extract_resposta_llama(resp)
    new_history = state.chat_history
    new_history.append(AIMessage(fResp))
    return {"answer": fResp, "chat_history": new_history}


# @monitor_recursos
def retrieve_docs_similarity_threshold_node(state: GraphState, vectorstore: FAISS, EMBEDDING_MODEL: str, threshold: float = 0.5,  k_docs: int = 4,fetch_k: int = 30) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    search_query = state.question
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device':'cpu'})

    q_vector = embedder.embed_query(search_query)
    new_queries = []
    prompts_queries = [PROMPT_QUERY_ASSUNTO,PROMPT_QUERY_OBJETIVO,PROMPT_QUERY_PONTOS_PRINCIPAIS, PROMPT_QUERY_CONTEUDO_RESPOSTA, PROMPT_QUERY_INFORMACOES_RELEVANTES_SOBRE]
    for prompt in prompts_queries:
        new_queries.append(generate_new_query(prompt=prompt, model_id=state.model_id,original_question=search_query))


    all_queries = [search_query] + new_queries 

    all_retrieved_docs = []
    q_vectors = []
    for query in all_queries:
        print("\n\nEmbedding...\n\n")
        q_vectors.append(embedder.embed_query(query))
    
    num_docs_per_query = k_docs // 5
    print("\n\nStarting retrieval...\n\n")
    for q_vector in q_vectors:

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
    print("Finished retrieval.\n\n")
    proc_docs = []
    for doc in docs:
        # print(doc)
        titulo = doc[0].metadata.get("titulo", "Sem título")
        arquivo = doc[0].metadata.get("arquivo_origem", "")
        # print(f"Arquivo {arquivo}. \n")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc[0].page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"source": url}))
    
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


# @monitor_recursos
def retrieve_docs_mmr_node(state: GraphState, vectorstore: FAISS, EMBEDDING_MODEL: str, k_docs: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5) -> Dict[str, Any]:
    """
    Nó que recupera os k arquivos do FAISS usando a pergunta atual e o histórico do chat.
    Utiliza o método as_retriever() para a busca.
    """
    search_query = state.question

    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device':'cpu'})

    q_vector = embedder.embed_query(search_query)
    new_queries = []
    prompts_queries = [PROMPT_QUERY_ASSUNTO,PROMPT_QUERY_OBJETIVO,PROMPT_QUERY_PONTOS_PRINCIPAIS, PROMPT_QUERY_CONTEUDO_RESPOSTA, PROMPT_QUERY_INFORMACOES_RELEVANTES_SOBRE]
    for prompt in prompts_queries:
        new_queries.append(generate_new_query(prompt=prompt, model_id=state.model_id,original_question=search_query))


    all_queries = [search_query] + new_queries 

    all_retrieved_docs = []
    q_vectors = []
    for query in all_queries:
        print("\n\nEmbedding...\n\n")
        q_vectors.append(embedder.embed_query(query))
    
    num_docs_per_query = k_docs // 5
    print("\n\nStarting retrieval...\n\n")
    for q_vector in q_vectors:
    

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
        unique_docs[doc[0].page_content] = doc

    docs = list(unique_docs.values())
    
    print("Retrieval finished.\n\n")
    proc_docs = []
    for doc in docs:
        # print(doc)
        titulo = doc[0].metadata.get("titulo", "Sem título")
        arquivo = doc[0].metadata.get("arquivo_origem", "")
        # print(f"Arq {arquivo}\n")
        url = f"https://sistemas.epagri.sc.gov.br/sedimob/consulta.action?subFuncao=consultaDiagnostico&cdEstrutura={arquivo.replace('.txt','')}&isEdicao=N&epagriTEC=S"
        descricao = _internal_extrair_descricao(doc[0].page_content)
        full_content = f"Título: {titulo}\n\nDescrição: {descricao}"
        proc_docs.append(Document(page_content=full_content, metadata={"source": url}))
    
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

# @monitor_recursos
def reset_answer(state: GraphState):
    return {"answer": "", "documents": [], "sources": [],"search_method": "similarity_threshold", "gen_lock": False}

# @monitor_recursos
def resume_historico(state: GraphState):

    ##Montar o prompt
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
    # flag = True
    # embedder_hf = HuggingFaceEmbeddings(model=EMBEDDING_MODEL,model_kwargs={ 'device':'cpu'})
    # embedder = LangchainEmbeddingsWrapper(embeddings=embedder_hf)
    # while flag:
    with torch.no_grad():
        resp = chain.invoke({
            "historico": formatted_hist
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    
    if state.model_id=="gemma2:9b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision-11b":
        fResp = _internal_extract_resposta_llama(resp)

        # data = []
        # data.append(
        #     {
        #         "reference_contexts": [formatted_hist],
        #         "response":fResp
        #     }
        # )
        # eval_data = EvaluationDataset.from_list(data)
        # evaluator_llm = LangchainLLMWrapper(langchain_llm=chat)
        # runconfig = RunConfig(max_workers=1)

        # try:
        #     results = evaluate(
        #         dataset=eval_data, 
        #         metrics=[
        #             SummarizationScore(llm=evaluator_llm)
        #         ],
        #         llm=evaluator_llm, 
        #         embeddings=embedder, 
        #         run_config=runconfig
        #     )
        #     print(results)
        # except Exception as e:
        #     print(f"    -> ERRO RAGAs: Falha ao executar 'evaluate': {e}")
        #     results = {}

        # print(fResp)
    new_history = []
    new_history.append(AIMessage(fResp))
    return {"chat_history": new_history}

# @monitor_recursos
def verify_history(state: GraphState):
    # print(f"\n\nTam chat_history: {len(state.chat_history)}\n\n")
    if len(state.chat_history) >= 1:
        return "yes"
    else:
        return "no"
    
# @monitor_recursos
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
    if state.model_id=="gemma2:9b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision-11b":
        fResp = _internal_extract_resposta_llama(resp)
    # print(f"\n\nVERIFY CONTEXT: {fResp}\n\n")
    if "não" in fResp: 
        return {"chat_history": []}
    else:
        return {"chat_history": context}
    

# @monitor_recursos
def generate_answer_without_history(state:GraphState):
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
    if state.model_id=="gemma2:9b":
        fResp = _internal_extract_resposta_gemma(resp)
    elif state.model_id=="myllama32-vision-11b":
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








# def search_duckduck(state: GraphState):
#     pergunta = state.question

#     ferramenta = DuckDuckGoSearchResults()

#     results = ferramenta.invoke(f"site:https://sistemas.epagri.sc.gov.br/sedimob/ {pergunta}")

#     print(results)