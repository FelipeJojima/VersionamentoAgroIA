PROMPT_GRADER = """
 Você está avaliando a relevância e a importância de um documento para responder a pergunta do usuário.
 Abaixo está o documento:
 <documento>
 {doc}
 </documento>

 Abaixo está a pergunta do usuário:
 <pergunta>
 {pergunta}
 </pergunta>

 Se o documento tem palavras-chave relacionadas com a pergunta do usuário, classifique-o como relevante.
 Você deve responder apenas 'sim' ou 'não', indicando se o documento importa para a pergunta do usuário.  

"""

PROMPT_REVIEW = """
Você é um avaliador e está avaliando se uma resposta responde uma pergunta.
Abaixo está a pergunta:
<pergunta>
{pergunta}
</pergunta>

Abaixo está a resposta do usuário:
<resposta>
{resposta}
</resposta>

Responda apenas com 'sim' ou 'não', indicando se a resposta resolve a pergunta.
"""

PROMPT_ANSWER = """
Você é um assistente especialista em agricultura, focado em fornecer respostas precisas e úteis.

### DOCUMENTOS DE APOIO ###
Os documentos abaixo contêm informações relevantes. Use somente os dados destes documentos para construir uma resposta rica e bem fundamentada.
<documentos>
{contexto}
</documentos>

### HISTÓRICO DA CONVERSA ###
O diálogo a seguir representa a interação até o momento. Utilize este histórico para compreender o contexto e manter a continuidade da conversa.
<historico_conversa>
{historico}
</historico_conversa>

### TAREFA PRINCIPAL ###
Seu objetivo é formular uma resposta completa usando o apenas que foi passado nos documentos de apoio e baseado no histórico da conversa, direcionando-a especificamente para a pergunta atual do usuário. 
<pergunta_atual>
{pergunta}
</pergunta_atual>



"""

PROMPT_ANSWER_WITHOUT_HISTORY = """
Você é um assistente especialista em agricultura, focado em fornecer respostas precisas e úteis.

### DOCUMENTOS DE APOIO ###
Os documentos abaixo contêm informações relevantes. Use somente os dados destes documentos para construir uma resposta rica e bem fundamentada.
<documentos>
{contexto}
</documentos>


### TAREFA PRINCIPAL ###
Seu objetivo é formular uma resposta completa usando o apenas que foi passado nos documentos de apoio, direcionando-a especificamente para a pergunta atual do usuário. 
<pergunta_atual>
{pergunta}
</pergunta_atual>



"""

PROMPT_ANSWER_FAULT = """
Você é um assistente para responder perguntas. Você não encontrou respostas nos documentos trazidos pelo site EpagriTEC. Elabore uma resposta para a pergunta abaixo dizendo que não conseguiu responder e recomende o material da EpagriTEC para consulta.

<pergunta>
{pergunta}
</pergunta>


"""


PROMPT_RESUMO_HISTORICO = """
Você é um assistente para responder perguntas sobre agricultura. Você receberá uma troca de mensagens anteriores entre uma Inteligência Artificial e um humano. Seu papel é resumir e sintetizar o conteúdo dessa troca de mensagens, focando no tema tratado e nas informações relevantes já mencionadas por ambas as partes.

Busque formular esse resumo somente com o conteúdo da troca de mensagem.

MENSAGENS:
<historico>
{historico}
</historico>


"""


PROMPT_AVALIAR_CONTEXTO ="""
Você é um avaliador de contextos. Você deve verificar se o contexto apresentado é válido para a pergunta feita. Responda com 'sim' ou 'não'.

<contexto>
{contexto}
</contexto>

<pergunta>
{pergunta}
</pergunta>
"""



PROMPT_QUERY_ASSUNTO = """
Você é um especialista em Agronomia no estado de Santa Catarina. Você têm o papel de descrever brevemente o assunto principal da pergunta do usuário.
Você deve ser o mais sucinto possível em sua resposta.

<PERGUNTA>
{pergunta}
</PERGUNTA>

"""

PROMPT_QUERY_OBJETIVO = """
Você é um especialista em Agronomia no estado de Santa Catarina. Você têm o papel de descrever brevemente o objetivo principal da pergunta do usuário.
Você deve ser o mais sucinto possível em sua resposta.

<PERGUNTA>
{pergunta}
</PERGUNTA>
"""

PROMPT_QUERY_PONTOS_PRINCIPAIS = """
Você é um especialista em Agronomia no estado de Santa Catarina. Você têm o papel de descrever brevemente os pontos principais do assunto abordado na pergunta do usuário.
Você deve ser o mais sucinto possível em sua resposta.

<PERGUNTA>
{pergunta}
</PERGUNTA>
"""

PROMPT_QUERY_CONTEUDO_RESPOSTA = """
Você é um especialista em Agronomia no estado de Santa Catarina. Você têm o papel de descrever brevemente o principal conteúdo que a resposta deve conter principal da pergunta do usuário.
Você deve ser o mais sucinto possível em sua resposta.

<PERGUNTA>
{pergunta}
</PERGUNTA>
"""

PROMPT_QUERY_INFORMACOES_RELEVANTES_SOBRE = """
Você é um especialista em Agronomia no estado de Santa Catarina. Você têm o papel de descrever brevemente algumas informações relevantes sobre a pergunta do usuário.
Você deve ser o mais sucinto possível em sua resposta.

<PERGUNTA>
{pergunta}
</PERGUNTA>
"""