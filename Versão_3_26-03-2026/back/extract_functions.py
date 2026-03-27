import re

# Para Modelos Llama
def _internal_extract_resposta_llama(resposta: str) -> str:
    padrao = r"(?:assistant\s*<\|end_header_id\|>\s*|assistant\n)(.*)"
    res = re.search(padrao, resposta, re.DOTALL)
    return res.group(1).strip() if res else resposta.strip()

def _internal_extract_resposta_gemma(resposta: str) -> str:
    padrao = r"(?:model\s+)(.*)"
    res = re.search(padrao, resposta, re.DOTALL)
    return res.group(1).strip() if res else resposta.strip()

def _internal_extrair_descricao(texto: str) -> str:
    padrao = r"(-- DESCRIÇÃO --)\s*(.+?)(\s*-- Informações Complementares --|$)"
    resultado = re.search(padrao, texto, re.DOTALL)
    if resultado:
        descricao = resultado.group(1).strip()
        # print(f"Desc: {descricao} \n")
        return descricao
    else:
        padrao =  r"\s*(.+?)(\s*-- Informações Complementares --|$)"
        resultado = re.search(padrao, texto, re.DOTALL)
        if resultado:
            descricao = resultado.group(1).strip()
            # print(f"Desc: {descricao} \n")
            return descricao
    return texto[:1000]