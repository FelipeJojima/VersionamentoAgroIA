from functools import partial

from GraphState_novo import GraphState, monitor_recursos, verify_lock_state, verify_gen_lock, gen_locker, gen_answer_state
from GraphState_novo import gen_answer_fault, generate_answer_node, generate_answer_without_history, generate_choice, gen_choice_state
from GraphState_novo import reset_answer, create_model_node, decide_to_generate, resume_historico, grading_node, review_answer_node, verify_history, verify_context
from GraphState_novo import retrieve_docs_similarity_node, retrieve_docs_similarity_threshold_node, retrieve_docs_mmr_node, gen_prev_hist, review_state
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from IPython.display import Image
from langchain_community.vectorstores import FAISS
from bert_score import BERTScorer
import tracemalloc
import sys
import uuid

import nest_asyncio
nest_asyncio.apply()

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
    retrieve_docs_with_args = partial(retrieve_docs_similarity_node,vectorstore=vectorstore,EMBEDDING_MODEL=EMBEDDING_MODEL, k_docs=k_docs,fetch_k=50)
    retrieve_docs_with_args_sim_threshold = partial(retrieve_docs_similarity_threshold_node,vectorstore=vectorstore,EMBEDDING_MODEL=EMBEDDING_MODEL, k_docs=k_docs, threshold=0.2,fetch_k=50)
    retrieve_docs_with_args_mmr = partial(retrieve_docs_mmr_node, vectorstore=vectorstore, EMBEDDING_MODEL=EMBEDDING_MODEL, k_docs=k_docs, fetch_k=50,lambda_mult=0.5)
    review_w_args = partial(review_answer_node, EMBEDDING_MODEL=EMBEDDING_MODEL)
    grading_w_args = partial(grading_node,EMBEDDING_MODEL=EMBEDDING_MODEL)


    workflow = StateGraph(GraphState)

    workflow.add_node("get_model", create_model_node)
    workflow.add_node("retrieve_docs_sim", retrieve_docs_with_args)
    workflow.add_node("grading", grading_w_args)
    workflow.add_node("generate_answer", generate_answer_node, defer=True)
    workflow.add_node("retrieve_docs_threshold", retrieve_docs_with_args_sim_threshold)
    workflow.add_node("retrieve_docs_mmr", retrieve_docs_with_args_mmr)
    workflow.add_node("reset_answer_field", reset_answer)
    workflow.add_node("gen_response_fault", gen_answer_fault)
    workflow.add_node("resume_history", resume_historico)
    workflow.add_node("gen_resp_without_hist", generate_answer_without_history, defer=True)
    workflow.add_node("verify_context", verify_context)
    workflow.add_node("choice_gen", gen_choice_state, defer=True)
    workflow.add_node("verify_lock", verify_lock_state)
    workflow.add_node("locker", gen_locker)
    workflow.add_node("answer_state", gen_answer_state)
    workflow.add_node("ver_hist", gen_prev_hist)
    workflow.add_node("review", review_state)


    workflow.add_edge(START, "get_model")
    workflow.add_edge("get_model", "reset_answer_field")
    workflow.add_edge("reset_answer_field", "retrieve_docs_threshold")
    workflow.add_conditional_edges(
        "reset_answer_field",
        verify_history,
        {"yes": "resume_history","no": "choice_gen"}
    )
    workflow.add_edge("resume_history", "verify_context")
    workflow.add_edge("verify_context", "choice_gen")

    workflow.add_edge("retrieve_docs_threshold", "grading")
    workflow.add_edge("grading", "choice_gen")

    workflow.add_conditional_edges(
        "choice_gen",
        generate_choice,
        {"gen_fault": "verify_lock", "gen_wout_hist": "verify_lock", "mmr": "retrieve_docs_mmr", "similarity": "retrieve_docs_sim", "generate": "verify_lock", "end":END}
    )

    workflow.add_conditional_edges(
        "verify_lock",
        verify_gen_lock,
        {"yes": END, "no": "answer_state"}
    )

    workflow.add_edge("retrieve_docs_mmr", "grading")
    workflow.add_edge("retrieve_docs_sim", "grading")

    workflow.add_conditional_edges(
        "answer_state",
        decide_to_generate,
        {"generate": "ver_hist", "no_docs_try_mmr": "retrieve_docs_mmr", "no_docs_try_sim": "retrieve_docs_sim", "no": "gen_response_fault"}
    )

    workflow.add_conditional_edges(
        "ver_hist",
        verify_history,
        {"yes": "generate_answer", "no":"gen_resp_without_hist"}
    )

    workflow.add_edge("generate_answer", "review")
    workflow.add_edge("gen_resp_without_hist", "review")

    workflow.add_conditional_edges(
        "review",
        review_w_args,
        {"não": "gen_response_fault", "end": END, "MMR": "retrieve_docs_mmr", "Similarity": "retrieve_docs_sim", "sim": END}
    )

    workflow.add_edge("gen_response_fault", END)


    ###Testes de busca
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
        
        print("Modelo e grafo carregados com sucesso!")
        try:
            img_data = RAG_APP.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
            with open("graph_final.png", "wb") as f:
                f.write(img_data)
            Image(img_data)
            # print(RAG_APP.get_graph().draw_mermaid())
            print("Grafo visual salvo como 'graph_final.png'")
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
            DEFAULT_LLM_MODEL = "gemma2:9b"
        case 2:
            DEFAULT_LLM_MODEL = "myllama32-vision-11b"

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
    # print("Sources (finals):\n{sources}")
    return {"answer": final_answer, "sources": sources, "history": history, "search_method": s_method, "question": pergunta}

def calculate_bert(num_model: int, resposta_modelo: str, ground_truth: str):
    if num_model == 0:
        BERT_MODEL_NAME = "FacebookAI/xlm-roberta-large"

        scorer = BERTScorer(
            model_type=BERT_MODEL_NAME,
            num_layers=17,
            lang="pt",
            rescale_with_baseline=False,
            device="cpu"
        )
    elif num_model == 1:
        BERT_MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
        scorer = BERTScorer(
            model_type=BERT_MODEL_NAME,
            num_layers=12,
            lang="pt",
            rescale_with_baseline=False,
            device="cpu"
        )        
    else:
        print("Insira um numero válido (0 ou 1) para modelo BertScorer!\n")
        return -1.0,-1.0,-1.0
    


    scorer._tokenizer.model_max_length = 512
    scorer._model.config.max_position_embeddings = 512 

    precision, recall, f1_score = scorer.score([resposta_modelo], [ground_truth])

    return precision, recall, f1_score






tracemalloc.start()

p1 = "Cite os principais cultivares da banana recomendados para o estado de Santa Catarina."
p2 = "Quais as principais características da cultivar SCS452 Nanicão Corupá?"
p3 = "Em relação ao MAPEAMENTO DAS ÁREAS PRODUTORAS DE BANANA EM SANTA CATARINA, quais as áreas colhidas para os subgrupos Cavendish e Prata e em quais municípios ocorrem as concentrações dessas colheitas?"
p5 = "Como deve-se proceder a coleta e amostragem de solo para uma área a ser implantada e uma em produção para o cultivo de banana em Santa Catarina?"
p8 = "Quais as principais etapas a serem realizadas para o manejo de um bananal?"
p9 = "Como é realizado o ensacamento de cachos e quais os seus benefícios?"
p10 = "Quais doenças não classificadas como quarentenárias na cultura da banana?"
p11 = "Descreva a antracnose na banana e seus principais sintomas"
p12 = "Quais as principais características do moleque da bananeira?"
p14 = " Quais as informações que um produtor deve saber sobre TEMPERATURA E UMIDADE RELATIVA PARA CLIMATIZAÇÃO da banana?"
perguntas = []
perguntas.append(p1)
perguntas.append(p2)
perguntas.append(p3)
perguntas.append(p5)
perguntas.append(p8)
perguntas.append(p9)
perguntas.append(p10)
perguntas.append(p11)
perguntas.append(p12)
perguntas.append(p14)

r1 = "Branca, BRS Fhia Maravilha, BRS Fhia Maravilha, BRS Platina, BRS Princesa, BRS SCS Belluna, BRS Thap Maeo, BRS Tropical, Figo cinza, Grande Naine, Nanicão, Ouro, Prata Anã, SCS451 Catarina, SCS452 Nanicão Corupá, SCS453 Noninha, SCS454 Carvoeira, Terra e Williams."
r2 = "Grupo genômico AAA, subgrupo Cavendish. Resultado de mutação natural originário do município de Corupá/SC. Tem porte baixo, boa produtividade e é altamente suscetível à sigatoka amarela e altamente resistente ao mal do panamá. As recomendações de espaçamento são de 2,5m x 2,5m; 2,0m x 3,0m, ou 1.600 plantas por hectare.",
r3 = "A área colhida de bananeiras do subgrupo Cavendish, distribuída pelo Estado com um total de 20.800ha, está concentração na região norte, em municípios como Corupá, com 4.830ha; Luiz Alves, com 3.900ha; e Massaranduba, com 1.957ha. Já o subgrupo Prata totaliza 7.500ha, havendo uma concentração dos principais municípios produtores no sul do Estado, com Jacinto Machado, representando 2.500ha, seguido de Santa Rosa do Sul, com 820ha, e Criciúma, com 560ha.",
r5 =     "Antes da implantação do pomar, devem-se coletar 20 amostras simples por área/gleba homogênea para formar a amostra composta que será analisada. As amostras devem ser coletadas de forma aleatória, cobrindo toda a área amostrada. Além da amostragem superficial (0-20cm), recomenda-se coletar, sempre que possível, amostras em subsuperfície (20-40cm), principalmente para caracterização da fertilidade do solo que antecede a implantação dos pomares e por ser o momento propício para incorporação de corretivos, condicionadores e fertilizantes em profundidade. Já a amostragem em pomares de bananeira em produção leva em consideração a forma de aplicação dos fertilizantes na superfície do solo. Quando a fertilização é realizada em área total, devem ser coletadas no mínimo 20 amostras simples de forma aleatória e bem distribuídas em toda a área do pomar. Por outro lado, nos pomares em que os fertilizantes são aplicados na superfície do solo de forma localizada, como em faixa, em meia-lua na frente da planta filha ou via fertirrigação, as amostras simples devem ser coletadas próximo à área fertilizada (a uma distância aproximada de 30cm da faixa de aplicação dos fertilizantes). Nesse caso, tendo em vista o gradiente de fertilidade e acidez do solo proporcionados pela adubação localizada “otimizada”, recomenda-se realizar a amostragem de solo também nas entrelinhas para monitorar a fertilidade do solo dessa área.",
r8 = "Controle de vegetação espontânea, Densidade de plantio, Desbaste de perfilhos, Desfollha, Desvio de filhotes e de cachos, Ensacamento dos cachos, Escoramento ou amarração da bananeira, Manejo do pseudocaule após a colheita, Poda de coração e Poda de pencas.",
r9 = "Pode ser feito de forma precoce, ou seja, em inflorescências pendentes e ainda não abertas, ou de forma tardia, quando os cachos apresentarem a falsa penca e duas a quatro pencas masculinas já abertas. Neste momento, o ensacamento pode ser feito concomitantemente com outras práticas como poda de pencas, poda do coração e escoramento, quando coincidentes. Para a execução manual do ensacamento, são necessários escada, sacos plásticos e tiras de fitilhos. Com o auxílio da escada, coloca-se o saco ao redor do cacho, prendendo com a tira de fitilho acima da cicatriz do engaço. O ensacamento também pode ser feito utilizando-se um equipamento específico denominado ensacador ou embolsador, indicado para fixação de sacos plásticos em cachos nas bananeiras. É uma prática essencial para a qualidade dos frutos, proporcionando as seguintes vantagens: aumento de peso dos cachos; produção de frutos mais longos e com maior diâmetro; encurtamento do período entre floração e colheita; produção de frutos com maior brilho; redução de danos mecânicos decorrentes do atrito causado pelas folhas, deposição de poeiras e ação dos ventos e granizo leve; controle de danos causados por diversos animais como insetos, roedores, pássaros, etc.; proteção contra a deposição de produtos químicos resultantes das pulverizações; redução de danos nas colheitas; e proteção dos frutos contra manchas de seiva.",
r10 = "MAL-DO-PANAMÁ – TR4, TOPO EM LEQUE, MOKO DA BANANEIRA e MOSAICO DAS BRÁCTEAS.",
r11 = "A antracnose é uma das principais doenças pós-colheita da bananeira e ocorre praticamente em todas as regiões produtoras. Essa doença prejudica a aparência e reduz o valor nutricional dos frutos, que ficam impróprios para o mercado. No Brasil, o único agente causal da antracnose é o fungo Colletotrichum musae, porém, existem relatos de diversas espécies de Colletotrichum capazes de causar doença em bananas. Os conídios de Colletotrichum musae produzidos em tecidos senescentes de bananeira, inclusive folhas, brácteas e pedúnculos, são liberados da matriz mucilaginosa que envolve os acérvulos e transportados para outros frutos por água livre, como chuva. O conídio se adere à superfície de um fruto verde, germina e forma apressório em até 72h. O fungo permanece quiescente até a maturação dos frutos, quando os sintomas podem ser visualizados. Colletotrichum musae leva a alterações fisiológicas e bioquímicas nos frutos que aceleram sua maturação e senescência. Porém, se a infecção ocorrer em ferimentos sobre frutos verdes, os sintomas podem se desenvolver. A doença se caracteriza pela formação de lesões escuras deprimidas restritas ao pericarpo de fruto maduro, exceto quando este é exposto a altas temperaturas, ou em adiantado estágio de maturação. Sob condições de alta umidade, o fungo produz acérvulos envoltos de uma matriz mucilaginosa alaranjada, que abriga os conídios do fungo.",
r12 = "Cosmopolites sordidus (Germar, 1824) - Ordem: Coleoptera - Família: Curculionoidae. Biologia: Besouros de coloração negra, medindo aproximadamente 11mm de comprimento por 5mm de largura. Apresentam hábitos de forrageamento e reprodução predominantemente noturnos, escondendo-se durante o dia em ambientes mais úmidos e abrigados da luz, como touceiras, bainhas e restos culturais do pomar. Apesar de possuir asas, esta espécie não voa. As fêmeas colocam seus ovos em pequenas cavidades no rizoma das plantas de bananeira. As larvas eclodem e penetram o tecido vegetal, desenvolvendo-se no interior dos rizomas por um período que varia entre 30 e 50 dias, dependendo dos níveis nutricionais da planta, do cultivar atacado, das condições climáticas e da temperatura. Após esse período, as larvas se dirigem mais uma vez para a periferia do rizoma, onde passam pelo processo de pupa, por um período de aproximadamente 10 dias, após o qual um novo adulto emerge.",
r14 = "A temperatura ideal para uma boa climatização é de 18°C para bananas do subgrupo Cavendish (Caturra) e 16°C para bananas do subgrupo Prata (Branca). No entanto, a climatização é possível numa faixa de 13°C até 20°C. Acima de 20°C, a maturação será acelerada e poderá diminuir a vida de prateleira no mercado e na mesa do consumidor. Acima de 21°C, poderá ocorrer o amolecimento da polpa, devido ao rápido processo de transformação do amido em açúcares, inviabilizando o transporte e a comercialização da fruta in natura. Abaixo de 12°C, acontece o chilling (queima por frio) na fruta. A casca fica com manchas esverdeadas e estrias escurecidas devido ao rompimento de células e vasos condutores, com extravasamento do líquido de seu interior. Quanto mais baixa a temperatura, até o limite inferior recomendado, maior será o tempo de climatização e mais longa a vida de prateleira do produto. A umidade relativa do ar dentro da câmara deve ficar entre 85% e 95%. Umidade acima de 95% pode levar ao desenvolvimento de doenças fúngicas de pós-colheita e o desverdecimento da casca é retardado. Para baixar a umidade dentro da câmara, utiliza-se aparelhos desumidificadores. Por outro lado, umidade abaixo de 85% pode causar perda de peso da fruta por desidratação, enrugamento da casca, desprendimento dos frutos da almofada, coloração opaca (sem brilho) da casca, retardamento da maturação e acentuação de manchas nas cascas. Para aumentar a umidade na câmara de climatização, podem-se utilizar nebulizadores de água, serragem molhada no piso ou recipientes contendo água."

respostas = []
respostas.append(r1)
respostas.append(r2)
respostas.append(r3)
respostas.append(r5)
respostas.append(r8)
respostas.append(r9)
respostas.append(r10)
respostas.append(r11)
respostas.append(r12)
respostas.append(r14)

initialize_model(4,25) #multilingual
thread_id=str(uuid.uuid4())
resp = search(pergunta=perguntas[int(sys.argv[1])-1], thread_id=thread_id, model=2)#llama
gt = respostas[int(sys.argv[1])-1]
print("Resposta do modelo:\n")
print(resp)

print("Ground-Truth:\n")
print(gt)

p_r, r_r, f1_r = calculate_bert(0, resp["answer"], gt)

p_t, r_t, f1_t = calculate_bert(1, resp["answer"], gt)

print("Roberta: Precision = " )
print(p_r) 
print("Recall = ")
print(r_r) 
print("F1 = ") 
print(f1_r) 
print("\n")



print("\n\nTimbau: Precision = " )
print(p_t) 
print("Recall = ")
print(r_t) 
print("F1 = ") 
print(f1_t) 
print("\n")