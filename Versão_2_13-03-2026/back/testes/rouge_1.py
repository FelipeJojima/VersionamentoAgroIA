# CÁLCULO DA MÉTRICA ROUGE 1

from rouge import Rouge
import json
import os
import sys

# Vetor preenchido com as respostas geradas pelo Rodrigo das questões: 1, 2, 3, 5, 8, 9, 10, 11, 12, 14, respectivamente
human_answer = [
    "Branca, BRS Fhia Maravilha, BRS Fhia Maravilha, BRS Platina, BRS Princesa, BRS SCS Belluna, BRS Thap Maeo, BRS Tropical, Figo cinza, Grande Naine, Nanicão, Ouro, Prata Anã, SCS451 Catarina, SCS452 Nanicão Corupá, SCS453 Noninha, SCS454 Carvoeira, Terra e Williams.",
    "Grupo genômico AAA, subgrupo Cavendish. Resultado de mutação natural originário do município de Corupá/SC. Tem porte baixo, boa produtividade e é altamente suscetível à sigatoka amarela e altamente resistente ao mal do panamá. As recomendações de espaçamento são de 2,5m x 2,5m; 2,0m x 3,0m, ou 1.600 plantas por hectare.",
    "A área colhida de bananeiras do subgrupo Cavendish, distribuída pelo Estado com um total de 20.800ha, está concentração na região norte, em municípios como Corupá, com 4.830ha; Luiz Alves, com 3.900ha; e Massaranduba, com 1.957ha. Já o subgrupo Prata totaliza 7.500ha, havendo uma concentração dos principais municípios produtores no sul do Estado, com Jacinto Machado, representando 2.500ha, seguido de Santa Rosa do Sul, com 820ha, e Criciúma, com 560ha.",
    "Antes da implantação do pomar, devem-se coletar 20 amostras simples por área/gleba homogênea para formar a amostra composta que será analisada. As amostras devem ser coletadas de forma aleatória, cobrindo toda a área amostrada. Além da amostragem superficial (0-20cm), recomenda-se coletar, sempre que possível, amostras em subsuperfície (20-40cm), principalmente para caracterização da fertilidade do solo que antecede a implantação dos pomares e por ser o momento propício para incorporação de corretivos, condicionadores e fertilizantes em profundidade. Já a amostragem em pomares de bananeira em produção leva em consideração a forma de aplicação dos fertilizantes na superfície do solo. Quando a fertilização é realizada em área total, devem ser coletadas no mínimo 20 amostras simples de forma aleatória e bem distribuídas em toda a área do pomar. Por outro lado, nos pomares em que os fertilizantes são aplicados na superfície do solo de forma localizada, como em faixa, em meia-lua na frente da planta filha ou via fertirrigação, as amostras simples devem ser coletadas próximo à área fertilizada (a uma distância aproximada de 30cm da faixa de aplicação dos fertilizantes). Nesse caso, tendo em vista o gradiente de fertilidade e acidez do solo proporcionados pela adubação localizada “otimizada”, recomenda-se realizar a amostragem de solo também nas entrelinhas para monitorar a fertilidade do solo dessa área.",
    "Controle de vegetação espontânea, Densidade de plantio, Desbaste de perfilhos, Desfollha, Desvio de filhotes e de cachos, Ensacamento dos cachos, Escoramento ou amarração da bananeira, Manejo do pseudocaule após a colheita, Poda de coração e Poda de pencas.",
    "Pode ser feito de forma precoce, ou seja, em inflorescências pendentes e ainda não abertas, ou de forma tardia, quando os cachos apresentarem a falsa penca e duas a quatro pencas masculinas já abertas. Neste momento, o ensacamento pode ser feito concomitantemente com outras práticas como poda de pencas, poda do coração e escoramento, quando coincidentes. Para a execução manual do ensacamento, são necessários escada, sacos plásticos e tiras de fitilhos. Com o auxílio da escada, coloca-se o saco ao redor do cacho, prendendo com a tira de fitilho acima da cicatriz do engaço. O ensacamento também pode ser feito utilizando-se um equipamento específico denominado ensacador ou embolsador, indicado para fixação de sacos plásticos em cachos nas bananeiras. É uma prática essencial para a qualidade dos frutos, proporcionando as seguintes vantagens: aumento de peso dos cachos; produção de frutos mais longos e com maior diâmetro; encurtamento do período entre floração e colheita; produção de frutos com maior brilho; redução de danos mecânicos decorrentes do atrito causado pelas folhas, deposição de poeiras e ação dos ventos e granizo leve; controle de danos causados por diversos animais como insetos, roedores, pássaros, etc.; proteção contra a deposição de produtos químicos resultantes das pulverizações; redução de danos nas colheitas; e proteção dos frutos contra manchas de seiva.",
    "MAL-DO-PANAMÁ – TR4, TOPO EM LEQUE, MOKO DA BANANEIRA e MOSAICO DAS BRÁCTEAS.",
    "A antracnose é uma das principais doenças pós-colheita da bananeira e ocorre praticamente em todas as regiões produtoras. Essa doença prejudica a aparência e reduz o valor nutricional dos frutos, que ficam impróprios para o mercado. No Brasil, o único agente causal da antracnose é o fungo Colletotrichum musae, porém, existem relatos de diversas espécies de Colletotrichum capazes de causar doença em bananas. Os conídios de Colletotrichum musae produzidos em tecidos senescentes de bananeira, inclusive folhas, brácteas e pedúnculos, são liberados da matriz mucilaginosa que envolve os acérvulos e transportados para outros frutos por água livre, como chuva. O conídio se adere à superfície de um fruto verde, germina e forma apressório em até 72h. O fungo permanece quiescente até a maturação dos frutos, quando os sintomas podem ser visualizados. Colletotrichum musae leva a alterações fisiológicas e bioquímicas nos frutos que aceleram sua maturação e senescência. Porém, se a infecção ocorrer em ferimentos sobre frutos verdes, os sintomas podem se desenvolver. A doença se caracteriza pela formação de lesões escuras deprimidas restritas ao pericarpo de fruto maduro, exceto quando este é exposto a altas temperaturas, ou em adiantado estágio de maturação. Sob condições de alta umidade, o fungo produz acérvulos envoltos de uma matriz mucilaginosa alaranjada, que abriga os conídios do fungo.",
    "Cosmopolites sordidus (Germar, 1824) - Ordem: Coleoptera - Família: Curculionoidae. Biologia: Besouros de coloração negra, medindo aproximadamente 11mm de comprimento por 5mm de largura. Apresentam hábitos de forrageamento e reprodução predominantemente noturnos, escondendo-se durante o dia em ambientes mais úmidos e abrigados da luz, como touceiras, bainhas e restos culturais do pomar. Apesar de possuir asas, esta espécie não voa. As fêmeas colocam seus ovos em pequenas cavidades no rizoma das plantas de bananeira. As larvas eclodem e penetram o tecido vegetal, desenvolvendo-se no interior dos rizomas por um período que varia entre 30 e 50 dias, dependendo dos níveis nutricionais da planta, do cultivar atacado, das condições climáticas e da temperatura. Após esse período, as larvas se dirigem mais uma vez para a periferia do rizoma, onde passam pelo processo de pupa, por um período de aproximadamente 10 dias, após o qual um novo adulto emerge.",
    "A temperatura ideal para uma boa climatização é de 18°C para bananas do subgrupo Cavendish (Caturra) e 16°C para bananas do subgrupo Prata (Branca). No entanto, a climatização é possível numa faixa de 13°C até 20°C. Acima de 20°C, a maturação será acelerada e poderá diminuir a vida de prateleira no mercado e na mesa do consumidor. Acima de 21°C, poderá ocorrer o amolecimento da polpa, devido ao rápido processo de transformação do amido em açúcares, inviabilizando o transporte e a comercialização da fruta in natura. Abaixo de 12°C, acontece o chilling (queima por frio) na fruta. A casca fica com manchas esverdeadas e estrias escurecidas devido ao rompimento de células e vasos condutores, com extravasamento do líquido de seu interior. Quanto mais baixa a temperatura, até o limite inferior recomendado, maior será o tempo de climatização e mais longa a vida de prateleira do produto. A umidade relativa do ar dentro da câmara deve ficar entre 85% e 95%. Umidade acima de 95% pode levar ao desenvolvimento de doenças fúngicas de pós-colheita e o desverdecimento da casca é retardado. Para baixar a umidade dentro da câmara, utiliza-se aparelhos desumidificadores. Por outro lado, umidade abaixo de 85% pode causar perda de peso da fruta por desidratação, enrugamento da casca, desprendimento dos frutos da almofada, coloração opaca (sem brilho) da casca, retardamento da maturação e acentuação de manchas nas cascas. Para aumentar a umidade na câmara de climatização, podem-se utilizar nebulizadores de água, serragem molhada no piso ou recipientes contendo água."
]


# Este vetor será preenchido a cada execução com o novo conjunto de 10 respostas do LLM
llm_answer = []

rouge = Rouge()

# Variáveis para armazenar o melhor resultado global ROUGE-1
best_f1_rouge_1 = -1.0
best_recall_rouge_1 = -1.0
best_precision_rouge_1 = -1.0

best_llm_answers_set = None
current_execution_number = 0

# --- ARQUIVOS GERADOS ---
BEST_RESULTS_FILE = "best_rouge_1_results.json" # Para o melhor resultado individual
ALL_EXECUTIONS_HISTORY_FILE = "rouge_1_executions_history.json" # Para o histórico de todas as execuções
AVERAGE_RESULTS_FILE = "rouge_1_average_results.json" # Para a média final das 30 execuções
QUESTION_MAP = [1, 2, 3, 5, 8, 9, 10, 11, 12, 14]
NUM_TOTAL_EXECUTIONS = 30

# --- FUNÇÕES PARA CARREGAR E SALVAR O MELHOR RESULTADO INDIVIDUAL ---
def load_best_results():
    """Carrega o melhor resultado ROUGE-1 salvo em um arquivo."""
    try:
        with open(BEST_RESULTS_FILE, 'r') as f:
            data = json.load(f)
            return data.get('best_f1_rouge_1', -1.0), \
                   data.get('best_recall_rouge_1', -1.0), \
                   data.get('best_precision_rouge_1', -1.0), \
                   data.get('best_llm_answers_set', None), \
                   data.get('current_execution_number_of_best', 0)
    except FileNotFoundError:
        return -1.0, -1.0, -1.0, None, 0

def save_best_results(f1, recall, precision, llm_answers, execution_num_of_best):
    """Salva o melhor resultado ROUGE-1 em um arquivo."""
    data = {
        'best_f1_rouge_1': f1,
        'best_recall_rouge_1': recall,
        'best_precision_rouge_1': precision,
        'best_llm_answers_set': llm_answers,
        'current_execution_number_of_best': execution_num_of_best
    }
    with open(BEST_RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

# --- FUNÇÕES PARA GERENCIAR O HISTÓRICO DE TODAS AS EXECUÇÕES ---
def load_execution_history():
    """Carrega o histórico de todas as execuções."""
    if os.path.exists(ALL_EXECUTIONS_HISTORY_FILE):
        with open(ALL_EXECUTIONS_HISTORY_FILE, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError: # Lida com arquivo vazio ou corrompido
                return {'executions': [], 'last_execution_number': 0}
    return {'executions': [], 'last_execution_number': 0}

def save_execution_history(history):
    """Salva o histórico de todas as execuções."""
    with open(ALL_EXECUTIONS_HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

# --- CARREGAR DADOS INICIAIS ---
best_f1_rouge_1, best_recall_rouge_1, best_precision_rouge_1, best_llm_answers_set, _ = load_best_results()
execution_history_data = load_execution_history()
current_execution_number = execution_history_data.get('last_execution_number', 0)

# Incrementa o número da execução atual
current_execution_number += 1
print(f"--- Execução Número: {current_execution_number} de {NUM_TOTAL_EXECUTIONS} ---")


### EXAMPLE ----------------------------------------------------------------------------
# Você precisará preencher 'llm_answer' aqui ANTES de cada execução do script.
# Exemplo (remova ou comente isso quando você realmente alimentar as respostas):
# llm_answer = [
#     "Alguns exemplos de cultivares de banana incluem Branca, BRS Fhia Maravilha, BRS Platina, BRS Princesa, BRS SCS Belluna, BRS Thap Maeo, BRS Tropical, Figo cinza, Grande Naine, Nanicão, Ouro, Prata Anã, SCS451 Catarina, SCS452 Nanicão Corupá, SCS453 Noninha, SCS454 Carvoeira, Terra e Williams.",
#     "A banana Nanicão Corupá é do grupo genômico AAA, subgrupo Cavendish. É uma mutação natural de Corupá/SC, com porte baixo, alta produtividade e alta suscetibilidade à sigatoka amarela, mas alta resistência ao mal do panamá. O espaçamento recomendado é de 2,5m x 2,5m; 2,0m x 3,0m, ou 1.600 plantas por hectare.",
#     "A área de cultivo de banana Cavendish em Santa Catarina totaliza 20.800ha, com maior concentração em Corupá (4.830ha), Luiz Alves (3.900ha) e Massaranduba (1.957ha). Já o subgrupo Prata totaliza 7.500ha, com os principais produtores no sul do Estado, como Jacinto Machado (2.500ha), Santa Rosa do Sul (820ha) e Criciúma (560ha).",
#     "Antes de plantar, colete 20 amostras simples por área homogênea para análise, de forma aleatória e abrangendo toda a área. Amostras superficiais (0-20cm) e de subsuperfície (20-40cm) são recomendadas para caracterizar a fertilidade do solo antes da implantação. Em pomares já em produção, se a fertilização é em área total, colete 20 amostras simples aleatoriamente. Se a fertilização é localizada (faixa, meia-lua, fertirrigação), colete amostras próximas à área fertilizada (30cm de distância) e também nas entrelinhas para monitorar a fertilidade.",
#     "As principais práticas culturais em bananeiras incluem: Controle de vegetação espontânea, Densidade de plantio, Desbaste de perfilhos, Desfollha, Desvio de filhotes e de cachos, Ensacamento dos cachos, Escoramento ou amarração da bananeira, Manejo do pseudocaule após a colheita, Poda de coração e Poda de pencas.",
#     "O ensacamento do cacho pode ser feito precocemente (inflorescências pendentes e não abertas) ou tardiamente (com falsa penca e 2-4 pencas masculinas abertas), podendo ser feito junto com poda de pencas, poda do coração e escoramento. Manualmente, usa-se escada, sacos plásticos e fitilhos, colocando o saco ao redor do cacho e prendendo acima da cicatriz do engaço. Há também o ensacador. As vantagens do ensacamento são: aumento de peso e tamanho dos frutos, encurtamento do período floração-colheita, maior brilho, redução de danos mecânicos decorrentes do atrito causado pelas folhas, deposição de poeiras e ação dos ventos e granizo leve; controle de danos causados por diversos animais como insetos, roedores, pássaros, etc.; proteção contra a deposição de produtos químicos resultantes das pulverizações; redução de danos nas colheitas; e proteção dos frutos contra manchas de seiva.",
#     "Doenças importantes da bananeira incluem: MAL-DO-PANAMÁ – TR4, TOPO EM LEQUE, MOKO DA BANANEIRA e MOSAICO DAS BRÁCTEAS.",
#     "A antracnose, causada pelo fungo Colletotrichum musae, é uma doença pós-colheita comum em bananeiras. Prejudica a aparência e o valor nutricional dos frutos, tornando-os impróprios para o mercado. Os conídios do fungo são liberados de tecidos senescentes e transportados pela água. Eles aderem ao fruto verde, germinam e formam apressórios, permanecendo quiescentes até a maturação. A doença causa lesões escuras deprimidas no pericarpo do fruto maduro. Em condições de alta umidade, o fungo produz acérvulos com conídios alaranjados.",
#     "Cosmopolites sordidus (Germar, 1824), um besouro preto de 11x5mm, é um curculionídeo. É noturno, escondendo-se de dia em locais úmidos. Não voa. As fêmeas põem ovos em pequenas cavidades no rizoma das plantas de bananeira. As larvas eclodem e penetram o tecido vegetal, desenvolvendo-se no interior dos rizomas por um período que varia entre 30 e 50 dias, dependendo dos níveis nutricionais da planta, do cultivar atacado, das condições climáticas e da temperatura. Após esse período, as larvas se dirigem mais uma vez para a periferia do rizoma, onde passam pelo processo de pupa, por um período de aproximadamente 10 dias, após o qual um novo adulto emerge.",
#     "A temperatura ideal para a climatização de bananas Cavendish é 18°C e para Prata é 16°C, com faixa possível de 13°C a 20°C. Acima de 20°C acelera a maturação e diminui a vida útil; acima de 21°C causa amolecimento da polpa, devido ao rápido processo de transformação do amido em açúcares, inviabilizando o transporte e a comercialização da fruta in natura. Abaixo de 12°C causa chilling (queima por frio), com manchas esverdeadas e estrias escuras na casca. Temperaturas mais baixas (dentro do limite) aumentam o tempo de climatização e a vida de prateleira. A umidade relativa do ar dentro da câmara deve ser entre 85% e 95%. Acima de 95% favorece doenças fúngicas e retarda o desverdecimento; abaixo de 85% causa perda de peso, enrugamento, desprendimento dos frutos, coloração opaca e retardo da maturação."
# ]
###




DIRETORIO_RESPOSTAS_LLM = f"/home/felipekenji/IC/IA/RAG/Epagri.ia-main/back/testes/after_scrap_testes/Exec {sys.argv[1]}"
ind_gemma_v2_answers = []
ind_llama_v2_answers = []
ind_gemma_v4_answers = []
ind_llama_v4_answers = []
seq_gemma_v2_answers = []
seq_gemma_v4_answers = []
seq_llama_v2_answers = []
seq_llama_v4_answers = []


all_resps = []

for filename in sorted(os.listdir(DIRETORIO_RESPOSTAS_LLM)):
    file_path = os.path.join(DIRETORIO_RESPOSTAS_LLM, filename)
    basename = os.path.splitext(filename)[0]

    #Identificacao da questão para ordenamento correto
    quest_number = 0
    if "p1" in basename:
        if "p10" in basename:
            quest_number = 7
        elif "p11" in basename:
            quest_number = 8
        elif "p12" in basename:
            quest_number = 9
        elif "p14" in basename:
            quest_number = 10
        else:
            quest_number = 1
    elif "p2" in basename:
        quest_number = 2
    elif "p3" in basename:
        quest_number = 3
    elif "p5" in basename:
        quest_number = 4
    elif "p8" in basename:
        quest_number = 5
    elif "p9" in basename:
        quest_number = 6


    #Identificacao do modelo
    switch_opt = 0
    if "ind" in basename:
        if "gemma" in basename:
            if "v2" in basename:
                switch_opt = 1
            else:
                switch_opt = 3
        else:
            if "v2" in basename:
                switch_opt = 2
            else:
                switch_opt = 4
    else:
        if "gemma" in basename:
            if "v2" in basename:
                switch_opt = 5
            else:
                switch_opt = 7
        else:
            if "v2" in basename:
                switch_opt = 6
            else:
                switch_opt = 8


    #Preenchimento dos vetores de respostas
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    match switch_opt:
        case 1:
            ind_gemma_v2_answers.insert((quest_number-1),content)
        case 2:
            ind_llama_v2_answers.insert((quest_number-1),content)
        case 3:
            ind_gemma_v4_answers.insert((quest_number-1),content)
        case 4:
            ind_llama_v4_answers.insert((quest_number-1),content)
        case 5:
            seq_gemma_v2_answers.insert((quest_number-1),content)
        case 6:
            seq_llama_v2_answers.insert((quest_number-1),content)
        case 7:
            seq_gemma_v4_answers.insert((quest_number-1),content)
        case 8:
            seq_llama_v4_answers.insert((quest_number-1),content)


all_resps.append(ind_gemma_v2_answers)
all_resps.append(ind_llama_v2_answers)
all_resps.append(ind_gemma_v4_answers)
all_resps.append(ind_llama_v4_answers)
all_resps.append(seq_gemma_v2_answers)
all_resps.append(seq_llama_v2_answers)
all_resps.append(seq_gemma_v4_answers)
all_resps.append(seq_llama_v4_answers)


num_comparisons = len(human_answer)

for indice,vetor in enumerate(all_resps):
    llm_answer.clear()
    llm_answer.extend(vetor)
    model = ""
    match indice:
        case 0:
            model = "ind_gemma_v2"
        case 1:
            model = "ind_llama_v2"
        case 2:
            model = "ind_gemma_v4"
        case 3:
            model = "ind_llama_v4"
        case 4:
            model = "seq_gemma_v2"
        case 5:
            model = "seq_llama_v2"
        case 6:
            model = "seq_gemma_v4"
        case 7:
            model = "seq_llama_v4"

    if not llm_answer: # Verifica se o vetor llm_answer está vazio
        print("O vetor 'llm_answer' está vazio. Por favor, preencha-o com as 10 respostas do LLM para esta execução.")
    elif num_comparisons != len(llm_answer):
        print("Os vetores human_answer e llm_answer devem ter o mesmo número de elementos (10).")
    else:
        rouge_1_recall_sum = 0
        rouge_1_precision_sum = 0
        rouge_1_f1_sum = 0

        current_individual_scores = []

        for i in range(num_comparisons):
            hypothesis = llm_answer[i]
            reference = human_answer[i]

            scores = rouge.get_scores(hypothesis, reference)

            # (MODIFICAÇÃO) Extrai scores ROUGE-L
            current_r = scores[0]['rouge-1']['r']
            current_p = scores[0]['rouge-1']['p']
            current_f = scores[0]['rouge-1']['f']

            # (MODIFICAÇÃO) Obtém o número da questão real
            question_number = QUESTION_MAP[i]

            # (MODIFICAÇÃO) Armazena os scores individuais com o número da questão
            current_individual_scores.append({
                'question_number': question_number,
                'rouge_1_recall': current_r,
                'rouge_1_precision': current_p,
                'rouge_1_f1': current_f
            })

            # Acumula scores para ROUGE-1
            rouge_1_recall_sum += scores[0]['rouge-1']['r']
            rouge_1_precision_sum += scores[0]['rouge-1']['p']
            rouge_1_f1_sum += scores[0]['rouge-1']['f']

        # Calcular as médias para ROUGE-1 da execução atual
        current_avg_rouge_1_recall = rouge_1_recall_sum / num_comparisons
        current_avg_rouge_1_precision = rouge_1_precision_sum / num_comparisons
        current_avg_rouge_1_f1 = rouge_1_f1_sum / num_comparisons


        print("\n--- Resultados da Execução Atual ---")
        print(f"ROUGE-1:")
        print(f"  Recall Médio: {current_avg_rouge_1_recall:.4f}")
        print(f"  Precisão Média: {current_avg_rouge_1_precision:.4f}")
        print(f"  F1-Score Médio: {current_avg_rouge_1_f1:.4f}")
        print("-" * 30)

        # --- ATUALIZAR O HISTÓRICO DE EXECUÇÕES ---
        execution_history_data['executions'].append({
            'execution_number': current_execution_number, # Nota: Isso será o mesmo para todos os 8 modelos nesta execução
            'model': model,
            'rouge_1_recall': current_avg_rouge_1_recall,
            'rouge_1_precision': current_avg_rouge_1_precision,
            'rouge_1_f1': current_avg_rouge_1_f1,
            'individual_scores': current_individual_scores # <-- Adicionado aqui
        })
        execution_history_data['last_execution_number'] = current_execution_number
        save_execution_history(execution_history_data)
        print(f"Resultados da execução {current_execution_number} adicionados ao histórico.")

        # Verifica se a execução atual é a melhor até agora para ROUGE-1
        # Note: O '_' no retorno de load_best_results ignora o número da melhor execução anterior para evitar confusão aqui.
        # Usaremos current_execution_number no save_best_results.
        if current_avg_rouge_1_f1 > best_f1_rouge_1:
            best_f1_rouge_1 = current_avg_rouge_1_f1
            best_recall_rouge_1 = current_avg_rouge_1_recall
            best_precision_rouge_1 = current_avg_rouge_1_precision

            best_llm_answers_set = list(llm_answer) # Salva uma cópia do conjunto de respostas do LLM, pode ser passado para o Rodrigo esse vetor

            # Salva o melhor resultado individual em arquivo
            save_best_results(best_f1_rouge_1, best_recall_rouge_1, best_precision_rouge_1, best_llm_answers_set, f"Exec {current_execution_number} - Modelo {model}")
            print(f"\n>>> NOVO MELHOR RESULTADO INDIVIDUAL ENCONTRADO na Execução {current_execution_number}! Salvo em '{BEST_RESULTS_FILE}'. <<<")
        else:
            # Se a execução atual não foi a melhor, re-salva os dados do melhor anterior
            # para garantir que o current_execution_number_of_best seja mantido.
            prev_f1_1, prev_recall_1, prev_precision_1, prev_answers, prev_exec_num = load_best_results()
            save_best_results(prev_f1_1, prev_recall_1, prev_precision_1, prev_answers, prev_exec_num)
            print(f"\nNão foi o melhor resultado individual. O melhor ROUGE-1 F1 continua sendo {best_f1_rouge_1:.4f} (da Execução {load_best_results()[4]})")


        print("\n--- Melhor Resultado ROUGE-1 até agora ---")
        print(f"Melhor F1-Score (ROUGE-1): {best_f1_rouge_1:.4f} (da Execução {load_best_results()[4]})")
        print(f"  Recall Associado: {best_recall_rouge_1:.4f}")
        print(f"  Precisão Associada: {best_precision_rouge_1:.4f}")

        # --- CALCULAR E SALVAR AS MÉDIAS FINAIS APÓS TODAS AS EXECUÇÕES ---
        if current_execution_number == NUM_TOTAL_EXECUTIONS:
            print(f"\n--- FIM DAS {NUM_TOTAL_EXECUTIONS} EXECUÇÕES. CALCULANDO MÉDIAS FINAIS ---")

            total_recall = sum(exec['rouge_1_recall'] for exec in execution_history_data['executions'])
            total_precision = sum(exec['rouge_1_precision'] for exec in execution_history_data['executions'])
            total_f1 = sum(exec['rouge_1_f1'] for exec in execution_history_data['executions'])

            avg_recall_30_executions = total_recall / NUM_TOTAL_EXECUTIONS
            avg_precision_30_executions = total_precision / NUM_TOTAL_EXECUTIONS
            avg_f1_30_executions = total_f1 / NUM_TOTAL_EXECUTIONS

            final_average_results = {
                'num_executions': NUM_TOTAL_EXECUTIONS,
                'average_rouge_1_recall': avg_recall_30_executions,
                'average_rouge_1_precision': avg_precision_30_executions,
                'average_rouge_1_f1': avg_f1_30_executions
            }

            with open(AVERAGE_RESULTS_FILE, 'w') as f:
                json.dump(final_average_results, f, indent=4)

            print(f"\nMédias das {NUM_TOTAL_EXECUTIONS} execuções (ROUGE-1) salvas em '{AVERAGE_RESULTS_FILE}':")
            print(f"  Recall Médio Global: {avg_recall_30_executions:.4f}")
            print(f"  Precisão Média Global: {avg_precision_30_executions:.4f}")
            print(f"  F1-Score Médio Global: {avg_f1_30_executions:.4f}")

        elif current_execution_number < NUM_TOTAL_EXECUTIONS:
            remaining = NUM_TOTAL_EXECUTIONS - current_execution_number
            print(f"Faltam {remaining} execuções para calcular a média final das {NUM_TOTAL_EXECUTIONS} rodadas.")