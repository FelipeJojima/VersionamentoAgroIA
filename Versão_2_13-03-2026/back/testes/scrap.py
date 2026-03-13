import os


texts_to_scrap_dir = "/home/felipekenji/IC/IA/RAG/Epagri.ia-main/back/testes/results_testes/Exec 3"

flag = 0

dest_dir = "after_scrap_testes"

os.makedirs(dest_dir, exist_ok=True)

for filename in sorted(os.listdir(texts_to_scrap_dir)):
    txt_path = os.path.join(texts_to_scrap_dir, filename)
    base_name = os.path.splitext(filename)[0]
    final_file = os.path.join(dest_dir, f"{base_name}.txt")
    lines = []
    try: #Pra tirar os zips da execução
        with open(txt_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        with open(txt_path, "r", encoding="latin-1") as file:
            lines = file.readlines()
    for numero_linha, linha in enumerate(lines, start=1):
        if "Sources:" in linha:
            flag = 0
        if flag==1:
            with open(final_file,"a",encoding="utf-8") as out_file:
                out_file.write(linha)
        if "Answer:" in linha:
            flag = 1 
    flag = 0