import os
import json


filename = "bests_exec.txt"

final_dir = "melhores_resultados"

origin_file = "rouge_l_executions_history.json"

with open(origin_file,"r") as f:
    data = json.load(f)

melhor_exec_ind_gemma_v2 = 0
melhor_exec_ind_llama_v2 = 0
melhor_exec_ind_gemma_v4 = 0
melhor_exec_ind_llama_v4 = 0
melhor_exec_seq_gemma_v2 = 0
melhor_exec_seq_llama_v2 = 0
melhor_exec_seq_gemma_v4 = 0
melhor_exec_seq_llama_v4 = 0

melhor_f1_ind_gemma_v2 = 0
melhor_f1_ind_llama_v2 = 0
melhor_f1_ind_gemma_v4 = 0
melhor_f1_ind_llama_v4 = 0
melhor_f1_seq_gemma_v2 = 0
melhor_f1_seq_llama_v2 = 0
melhor_f1_seq_gemma_v4 = 0
melhor_f1_seq_llama_v4 = 0



# print(data)
for i in data['executions']:
    if i['model']=="ind_gemma_v2":
        if i['rouge_l_f1']>melhor_f1_ind_gemma_v2:
            melhor_f1_ind_gemma_v2 = i['rouge_l_f1']
            melhor_exec_ind_gemma_v2 = i['execution_number']

    elif i['model']=="ind_llama_v2":
        if i['rouge_l_f1']>melhor_f1_ind_llama_v2:
            melhor_f1_ind_llama_v2 = i['rouge_l_f1']
            melhor_exec_ind_llama_v2 = i['execution_number']

    elif i['model']=="ind_gemma_v4":
        if i['rouge_l_f1']>melhor_f1_ind_gemma_v4:
            melhor_f1_ind_gemma_v4 = i['rouge_l_f1']
            melhor_exec_ind_gemma_v4 = i['execution_number']

    elif i['model']=="ind_llama_v4":
        if i['rouge_l_f1']>melhor_f1_ind_llama_v4:
            melhor_f1_ind_llama_v4= i['rouge_l_f1']
            melhor_exec_ind_llama_v4 = i['execution_number']

    elif i['model']=="seq_gemma_v2":
        if i['rouge_l_f1']>melhor_f1_seq_gemma_v2:
            melhor_f1_seq_gemma_v2 = i['rouge_l_f1']
            melhor_exec_seq_gemma_v2 = i['execution_number']

    elif i['model']=="seq_llama_v2":
        if i['rouge_l_f1']>melhor_f1_seq_llama_v2:
            melhor_f1_seq_llama_v2 = i['rouge_l_f1']
            melhor_exec_seq_llama_v2 = i['execution_number']

    elif i['model']=="seq_gemma_v4":
        if i['rouge_l_f1']>melhor_f1_seq_gemma_v4:
            melhor_f1_seq_gemma_v4 = i['rouge_l_f1']
            melhor_exec_seq_gemma_v4 = i['execution_number']

    elif i['model']=="seq_llama_v4":
        if i['rouge_l_f1']>melhor_f1_seq_llama_v4:
            melhor_f1_seq_llama_v4 = i['rouge_l_f1']
            melhor_exec_seq_llama_v4 = i['execution_number']

with open(f"{final_dir}/INFO.txt","w+",encoding="utf-8") as info_file:
    info_file.write(f"melhor_exec_ind_gemma_v2: {melhor_exec_ind_gemma_v2}\n")
    info_file.write(f"melhor_exec_ind_llama_v2: {melhor_exec_ind_llama_v2}\n")
    info_file.write(f"melhor_exec_ind_gemma_v4: {melhor_exec_ind_gemma_v4}\n")
    info_file.write(f"melhor_exec_ind_llama_v4: {melhor_exec_ind_llama_v4}\n")
    info_file.write(f"melhor_exec_seq_gemma_v2: {melhor_exec_seq_gemma_v2}\n")
    info_file.write(f"melhor_exec_seq_llama_v2: {melhor_exec_seq_llama_v2}\n")
    info_file.write(f"melhor_exec_seq_gemma_v4: {melhor_exec_seq_gemma_v4}\n")
    info_file.write(f"melhor_exec_seq_llama_v4: {melhor_exec_seq_llama_v4}\n")



iterador = []
iterador.append((melhor_exec_ind_gemma_v2,"ind_gemma_faiss_v2"))
iterador.append((melhor_exec_ind_llama_v2,"ind_llama_faiss_v2"))
iterador.append((melhor_exec_ind_gemma_v4,"ind_gemma_faiss_v4"))
iterador.append((melhor_exec_ind_llama_v4,"ind_llama_faiss_v4"))
iterador.append((melhor_exec_seq_gemma_v2,"seq_gemma_faiss_v2"))
iterador.append((melhor_exec_seq_llama_v2,"seq_llama_faiss_v2"))
iterador.append((melhor_exec_seq_gemma_v4,"seq_gemma_faiss_v4"))
iterador.append((melhor_exec_seq_llama_v4,"seq_llama_faiss_v4"))

resps_ind_gemma_v2 = [None] * 10
resps_ind_llama_v2 = [None] * 10
resps_ind_gemma_v4 = [None] * 10
resps_ind_llama_v4 = [None] * 10
resps_seq_gemma_v2 = [None] * 10
resps_seq_llama_v2 = [None] * 10
resps_seq_gemma_v4 = [None] * 10
resps_seq_llama_v4 = [None] * 10

for exec_number,name in iterador:
    resps_dir = f"/home/felipekenji/IC/IA/RAG/Epagri.ia-main/back/testes/after_scrap_testes/Exec {exec_number}"

    for file in os.listdir(resps_dir):
        file_path = os.path.join(resps_dir, file)
        basename = os.path.splitext(file)[0]
        # print(basename)

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

        if name in basename:
            with open(file_path,"r",encoding="utf-8") as resposta:
                content = resposta.read()
            match name:
                case "ind_gemma_faiss_v2":
                    resps_ind_gemma_v2[quest_number - 1] = content
                case "ind_llama_faiss_v2":
                    resps_ind_llama_v2[quest_number - 1] = content
                case "ind_gemma_faiss_v4":
                    resps_ind_gemma_v4[quest_number - 1] = content
                case "ind_llama_faiss_v4":
                    resps_ind_llama_v4[quest_number - 1] = content
                case "seq_gemma_faiss_v2":
                    resps_seq_gemma_v2[quest_number - 1] = content
                case "seq_llama_faiss_v2":
                    resps_seq_llama_v2[quest_number - 1] = content
                case "seq_gemma_faiss_v4":
                    resps_seq_gemma_v4[quest_number - 1] = content
                case "seq_llama_faiss_v4":
                    resps_seq_llama_v4[quest_number - 1] = content
                    
for n,name in iterador:
    match name:
        case "ind_gemma_faiss_v2":
            final_file = "ind_gemma_v2"
            with open(f"{final_dir}/{final_file}.txt", "w+",encoding="utf-8") as final:
                for i,j in enumerate(resps_ind_gemma_v2):
                    if i<3:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+1}\n")
                    elif i==3:
                        final.write(f"\n-------------------------------------------\n\nPergunta 5\n")
                    elif i>3 and i<9:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+4}\n")
                    else:
                        final.write(f"\n-------------------------------------------\n\nPergunta 14\n")
                    final.write(j)

        case "ind_llama_faiss_v2":
            final_file = "ind_llama_v2"
            with open(f"{final_dir}/{final_file}.txt", "w+",encoding="utf-8") as final:
                for i,j in enumerate(resps_ind_llama_v2):
                    if i<3:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+1}\n")
                    elif i==3:
                        final.write(f"\n-------------------------------------------\n\nPergunta 5\n")
                    elif i>3 and i<9:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+4}\n")
                    else:
                        final.write(f"\n-------------------------------------------\n\nPergunta 14\n")
                    final.write(j)

        case "ind_gemma_faiss_v4":
            final_file = "ind_gemma_v4"
            with open(f"{final_dir}/{final_file}.txt", "w+",encoding="utf-8") as final:
                for i,j in enumerate(resps_ind_gemma_v4):
                    if i<3:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+1}\n")
                    elif i==3:
                        final.write(f"\n-------------------------------------------\n\nPergunta 5\n")
                    elif i>3 and i<9:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+4}\n")
                    else:
                        final.write(f"\n-------------------------------------------\n\nPergunta 14\n")
                    final.write(j)

        case "ind_llama_faiss_v4":
            final_file = "ind_llama_v4"
            with open(f"{final_dir}/{final_file}.txt", "w+",encoding="utf-8") as final:
                for i,j in enumerate(resps_ind_llama_v4):
                    if i<3:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+1}\n")
                    elif i==3:
                        final.write(f"\n-------------------------------------------\n\nPergunta 5\n")
                    elif i>3 and i<9:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+4}\n")
                    else:
                        final.write(f"\n-------------------------------------------\n\nPergunta 14\n")
                    final.write(j)

        case "seq_gemma_faiss_v2":
            final_file = "seq_gemma_v2"
            with open(f"{final_dir}/{final_file}.txt", "w+",encoding="utf-8") as final:
                for i,j in enumerate(resps_seq_gemma_v2):
                    if i<3:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+1}\n")
                    elif i==3:
                        final.write(f"\n-------------------------------------------\n\nPergunta 5\n")
                    elif i>3 and i<9:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+4}\n")
                    else:
                        final.write(f"\n-------------------------------------------\n\nPergunta 14\n")
                    final.write(j)

        case "seq_llama_faiss_v2":
            final_file = "seq_llama_v2"
            with open(f"{final_dir}/{final_file}.txt", "w+",encoding="utf-8") as final:
                for i,j in enumerate(resps_seq_llama_v2):
                    if i<3:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+1}\n")
                    elif i==3:
                        final.write(f"\n-------------------------------------------\n\nPergunta 5\n")
                    elif i>3 and i<9:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+4}\n")
                    else:
                        final.write(f"\n-------------------------------------------\n\nPergunta 14\n")
                    final.write(j)

        case "seq_gemma_faiss_v4":
            final_file = "seq_gemma_v4"
            with open(f"{final_dir}/{final_file}.txt", "w+",encoding="utf-8") as final:
                for i,j in enumerate(resps_seq_gemma_v4):
                    if i<3:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+1}\n")
                    elif i==3:
                        final.write(f"\n-------------------------------------------\n\nPergunta 5\n")
                    elif i>3 and i<9:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+4}\n")
                    else:
                        final.write(f"\n-------------------------------------------\n\nPergunta 14\n")
                    final.write(j)

        case "seq_llama_faiss_v4":
            final_file = "seq_llama_v4"
            with open(f"{final_dir}/{final_file}.txt", "w+",encoding="utf-8") as final:
                for i,j in enumerate(resps_seq_llama_v4):
                    if i<3:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+1}\n")
                    elif i==3:
                        final.write(f"\n-------------------------------------------\n\nPergunta 5\n")
                    elif i>3 and i<9:
                        final.write(f"\n-------------------------------------------\n\nPergunta {i+4}\n")
                    else:
                        final.write(f"\n-------------------------------------------\n\nPergunta 14\n")
                    final.write(j)

