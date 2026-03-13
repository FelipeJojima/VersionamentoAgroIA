from model_for_tests import initialize_model, search
import uuid
import os
import sys
# from huggingface_hub import login
# login("--")


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

# models_embedder = [2,4] #2 = Faiss V2 - SENTENCE-TRANSFORMERS + CHUNKS// 4 = Faiss V4 - INT-FLOAT-MULTILINGUAL + CHUNKS
# models_chat = [1,2] # 1 - GEMMA// 2 - LLAMA

diretorio = "results_testes"
os.makedirs(diretorio, exist_ok=True)
thread_id=str(uuid.uuid4())

#5 A 6 ARGS SERAO PASSADOS NA LINHA DE CODIGO: 
# O PRIMEIRO É O TIPO DE EXEC (0 PARA INDIVIDUAL E 1 PARA SEQUENCIAL),
# O SEGUNDO É O EMBEDDER (2 OU 4), 
# O TERCEIRO É O MODELO (1 OU 2), 
# O QUARTO É O NUMERO DE DOCUMENTOS, 
# O QUINTO É O NUMERO DA EXECUCAO E 
# O ÚLTIMO É A PERGUNTA PARA O INDIVIDUAL (1 A 10)

emb_number = 0
model_number = 0
exec_number = int(sys.argv[5])
if int(sys.argv[2]) == 2:
    emb_number = 2
elif int(sys.argv[2]) == 4:
    emb_number = 4
else:
    print("Numero de embedder errado!")

if int(sys.argv[3])== 1:
    model_number = 1
elif int(sys.argv[3]) == 2:
    model_number = 2
else:
    print("Numero de modelo errado!")

num_docs = int(sys.argv[4])

print("Starting model with:\nNum_docs: " + num_docs.__str__() + "\nemb: " + emb_number.__str__() + "\nmodel: " + model_number.__str__() + "\n")
# initialize_model(emb_number,num_docs)

if int(sys.argv[1]) == 0:
    num_pergunta = int(sys.argv[6])
    name = ""
    resp = {}
    match num_pergunta:
        case 1:
            name = "p1"
            # resp = search(pergunta=perguntas[0],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
        case 2:
            name = "p2"
            # resp = search(pergunta=perguntas[1],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
        case 3:
            name = "p3"
            # resp = search(pergunta=perguntas[2],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
        case 4:
            name = "p5"
            # resp = search(pergunta=perguntas[3],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
        case 5:
            name = "p8"
            # resp = search(pergunta=perguntas[4],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
        case 6:
            name = "p9"
            # resp = search(pergunta=perguntas[5],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
        case 7:
            name = "p10"
            # resp = search(pergunta=perguntas[6],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
        case 8:
            name = "p11"
            # resp = search(pergunta=perguntas[7],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
        case 9:
            name = "p12"
            # resp = search(pergunta=perguntas[8],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
        case 10:
            name = "p14"
            # resp = search(pergunta=perguntas[9],thread_id=thread_id, model=model_number)
            print("Fazendo a " + name + "\n")
    if model_number == 1:
        file = f"{name}_ind_gemma_faiss_v{emb_number}_exec_{exec_number}.txt"
    elif model_number == 2:
        file = f"{name}_ind_llama_faiss_v{emb_number}_exec_{exec_number}.txt"
    print(file)
    # print(f"Resposta Final: {resp}.\n\nFinalizando...\n\n")
    # with open(f"{diretorio}/{file}","w+",encoding="utf-8") as f:
    #     f.write(resp.get("question") + "\n\nSearch Method: " + resp.get("search_method") + "\n\nChat history:\n" + resp.get("history") + "\n\nAnswer:\n" + resp.get("answer") + "\n\nSources:\n" + resp.get("sources") + "\n\n")
elif int(sys.argv[1]) == 1:
    for k, i in enumerate(perguntas):
        resp = {}
        # resp = search(i,thread_id, model_number)
        if model_number == 1:
            if k==3:
                arq_name = f"p5_seq_gemma_faiss_v{emb_number}_exec_{exec_number}.txt"
            elif k>=4 and k<9:
                arq_name = f"p{k+4}_seq_gemma_faiss_v{emb_number}_exec_{exec_number}.txt"
            elif k==9:
                arq_name = f"p14_seq_gemma_faiss_v{emb_number}_exec_{exec_number}.txt"
            else:
                arq_name = f"p{k+1}_seq_gemma_faiss_v{emb_number}_exec_{exec_number}.txt"
        elif model_number == 2:
            if k==3:
                arq_name = f"p5_seq_llama_faiss_v{emb_number}_exec_{exec_number}.txt"
            elif k>=4 and k<9:
                arq_name = f"p{k+4}_seq_llama_faiss_v{emb_number}_exec_{exec_number}.txt"
            elif k==9:
                arq_name = f"p14_seq_llama_faiss_v{emb_number}_exec_{exec_number}.txt"
            else:
                arq_name = f"p{k+1}_seq_llama_faiss_v{emb_number}_exec_{exec_number}.txt"
        print(arq_name)
        # print(f"Resposta Final: {resp}.\n\nFinalizando...\n\n")
        # with open(f"{diretorio}/{arq_name}","w+",encoding="utf-8") as f:
        #     f.write(resp.get("question") + "\n\nSearch Method: " + resp.get("search_method") + "\n\nChat history:\n" + resp.get("history") + "\n\nAnswer:\n" + resp.get("answer") + "\n\nSources:\n" + resp.get("sources") + "\n\n")
else: 
    print("Numero de modo errado!")