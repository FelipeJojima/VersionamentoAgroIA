import re

# Para Modelos Llama
# def _internal_extract_resposta(resposta: str) -> str:
#     padrao = r"(?:assistant\s*<\|end_header_id\|>\s*|assistant\n)(.*)"
#     res = re.search(padrao, resposta, re.DOTALL)
#     return res.group(1).strip() if res else resposta.strip()

def _internal_extract_resposta(resposta: str) -> str:
    padrao = r"(?:model\s+)(.*)"
    res = re.search(padrao, resposta, re.DOTALL)
    return res.group(1).strip() if res else resposta.strip()

def _internal_extrair_descricao(texto: str) -> str:
    padrao = r"-- DESCRIÇÃO --\s*(.+?)(\s*-- Informações Complementares --|$)"
    resultado = re.search(padrao, texto, re.DOTALL)
    if resultado:
        descricao = resultado.group(1).strip()
        return (descricao[:1000] + " [...]") if len(descricao) > 1000 else descricao
    return texto[:1000]