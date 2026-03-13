import json
import os


file = "best_rouge_l_results.json"

nFile = "best_resps.txt"



with open(file, "r") as f:
    data = json.load(f)
    

with open(nFile, "w",encoding="utf-8") as final_file:
    for ind, i in enumerate(data['best_llm_answers_set']):
        if ind<3:
            final_file.write(f"\n-------------------------------------------\n\nPergunta {ind+1}\n")
        elif ind==3:
            final_file.write(f"\n-------------------------------------------\n\nPergunta 5\n")
        elif ind>3 and ind<9:
            final_file.write(f"\n-------------------------------------------\n\nPergunta {ind+4}\n")
        else:
            final_file.write(f"\n-------------------------------------------\n\nPergunta 14\n")
        final_file.write(i)
