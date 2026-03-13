from model_for_tests_novo import initialize_model, search
import uuid
import sys
import os
from huggingface_hub import login
login("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")


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


initialize_model(4,30)
# dir_dest = "testes_3" 
# os.makedirs(dir_dest, exist_ok=True)
thread_id=str(uuid.uuid4())
print(search("doença quarentenária",thread_id,1))
# for k, i in enumerate(perguntas):
#     resp = search(i,thread_id, 1)
#     if k==3:
#         arq_name = f"p5_seq_gemma_faiss_v3_exec_{sys.argv[1]}.txt"
#     elif k>=4 and k<9:
#         arq_name = f"p{k+4}_seq_gemma_faiss_v3_exec_{sys.argv[1]}.txt"
#     elif k==9:
#         arq_name = f"p14_seq_gemma_faiss_v3_exec_{sys.argv[1]}.txt"
#     else:
#         arq_name = f"p{k+1}_seq_gemma_faiss_v3_exec_{sys.argv[1]}.txt"
#     with open(f"{dir_dest}/{arq_name}","w+",encoding="utf-8") as f:
#         f.write(resp.get("question") + "\n\nSearch Method: " + resp.get("search_method") + "\n\nChat history:\n" + resp.get("history") + "\n\nAnswer:\n" + resp.get("answer") + "\n\nSources:\n" + resp.get("sources") + "\n\n")