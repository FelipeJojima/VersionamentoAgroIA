# ==========================================================================
#         Worker que lida com as filas de processos - Epagri.IA
# ==========================================================================
# Este script representa um worker de segundo plano. Sua única função
# é esperar por tarefas computacionalmente pesadas (como rodar um modelo de IA)
# em filas do Redis, executá-las e publicar o resultado. Isso evita que a
# aplicação web principal (Flask) trave enquanto espera por uma resposta da IA
# ==========================================================================

import redis
import time
import json
import traceback
import torch
import whisper    # Modelo da OpenAI para transcrição de audio
import os
from back.model import initialize_model, search as search_model

# --- Definição dos Nomes das Filas e Listas no Redis ---
RAG_TASKS_QUEUE = "rag_tasks_queue"
TRANSCRIPTION_TASKS_QUEUE = "transcription_tasks_queue"
FINISHED_TASKS_LIST = "finished_tasks"

# Garante que o worker crie a pasta se ela não existir
TEMP_AUDIO_FOLDER = '/home/admin/Projects/Epagri_ia/back/temp_audio'
os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)

# ------------------------------------------------------ 1. CONFIGURAÇÃO E CONEXÃO COM O REDIS ------------------------------------------------------
# Tenta se conectar ao Redis. Se falhar, o worker não pode funcionar, entao o programa encerra
try:
    # Cria uma instância do cliente Redis
    # decode_responses=True é importante para que os dados lidos do Redis venham como string
    redis_conn = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_conn.ping()
    print("Conectado ao Redis com sucesso!")
except redis.exceptions.ConnectionError as e:
    print(f"Erro ao conectar ao Redis: {e}")
    print("Por favor, garanta que o serviço Redis está rodando.")
    exit(1)  # Encerra o script com um código de erro.

# -------------------------------------------------------- 2. INICIALIZAÇÃO DOS MODELOS DE IA -------------------------------------------------------
# Esta é uma etapa lenta e que consome muita memória (RAM e VRAM). Ela acontece apenas UMA VEZ quando o worker é iniciado, garantindo que as respostas subsequentes sejam rápidas
print("Iniciando o worker...")
print("Carregando o modelo de IA. Isso pode levar alguns minutos...")
try:
    # A função initialize_model (de back/model.py) carrega o modelo LLM
    initialize_model(4,20)
    print("Modelo carregado e pronto para receber perguntas.")
except Exception as e:
    print(f"Erro ao inicializar o modelo: {e}")
    traceback.print_exc()  # Imprime o erro detalhado
    exit(1)

try:
    # Carrega o modelo "base" do Whisper para transcrição de áudio
    whisper_model = whisper.load_model("base")
    print("Modelo Whisper carregado com sucesso.")
except Exception as e:
    print(f"Erro ao inicializar o modelo Whisper: {e}")
    traceback.print_exc()
    exit(1)


# -------------------------------------------------------- 3. LOOP DE PROCESSAMENTO PRINCIPAL ---------------------------------------------------------
def process_tasks():
    # Lista de filas que o worker vai escutar simultaneamente
    queues = [RAG_TASKS_QUEUE, TRANSCRIPTION_TASKS_QUEUE]
    print(f"Worker aguardando por tarefas nas filas: {queues}...")

    while True:
        job_id = None
        task_info_for_logging = {}

        try:
            # redis_conn.brpop é um comando de "pop bloqueante". O script irá pausar nesta linha até que uma tarefa apareça em uma das filas. timeout=0 significa esperar para sempre. Ele retorna o nome da fila e os dados da tarefa
            source_queue, task_data_json = redis_conn.brpop(queues, timeout=0)

            # Converte a string JSON da tarefa de volta para um dicionário Python
            task_data = json.loads(task_data_json)
            job_id = task_data.get("job_id")

            # Cria um canal de publicação único para esta tarefa, para que a resposta vá apenas para quem a pediu
            result_channel = f"results:{job_id}"

            # Tasks para responder perguntas
            if source_queue == RAG_TASKS_QUEUE:
                print(f"\n[+] Recebida tarefa de RAG (Job ID: {job_id})")
                pergunta = task_data.get("pergunta")
                thread_id = task_data.get("thread_id")
                task_info_for_logging = {"question": pergunta}  # Info para o log de tarefas.

                # Chama a função de busca do nosso módulo de IA, que faz todo o trabalho pesado.
                response_data = search_model(pergunta, thread_id,1)


                print(f"DEBUG: Tipo de response_data: {type(response_data)}")
                print(f"DEBUG: Conteúdo de response_data: {response_data}")

                # Converte o dicionário de resposta em uma string JSON para envio.
                result_payload = json.dumps(response_data)

                # Publica a resposta no canal específico do job. A aplicação Flask estará ouvindo.
                redis_conn.publish(result_channel, result_payload)
                print(f"[/] Resposta RAG para {job_id} publicada")

            # Tasks de transcricao de audio
            elif source_queue == TRANSCRIPTION_TASKS_QUEUE:
                print(f"\n[+] Recebida tarefa de Transcrição (Job ID: {job_id})")
                filepath = task_data.get("filepath")
                task_info_for_logging = {"question": f"Áudio: {os.path.basename(filepath)}"}

                # Usa o modelo Whisper para converter o áudio em texto.
                result = whisper_model.transcribe(filepath, language="pt")
                transcribed_text = result["text"]

                response_data = {"text": transcribed_text}
                result_payload = json.dumps(response_data)

                redis_conn.publish(result_channel, result_payload)
                print(f"[/] Transcrição para {job_id} publicada: '{transcribed_text}'")

                # Limpeza: apaga o arquivo de áudio temporário.
                if os.path.exists(filepath):
                    os.remove(filepath)

            # Após o sucesso, registra a tarefa em uma lista no Redis para fins de monitoramento.
            finished_record = json.dumps({
                "job_id": job_id,
                "question": task_info_for_logging.get("question", "N/A"),
                "finished_at": time.time()
            })

            # lpush adiciona o registro no topo da lista.
            redis_conn.lpush(FINISHED_TASKS_LIST, finished_record)
            # ltrim mantém a lista com no máximo 21 itens
            redis_conn.ltrim(FINISHED_TASKS_LIST, 0, 20)
            print(f"[*] Tarefa {job_id} registrada como finalizada.")

        except Exception as e:
            print(f"\n[!] Ocorreu um erro ao processar a tarefa (Job ID: {job_id}):")
            traceback.print_exc()
            if job_id:
                error_payload = json.dumps({"error": "Ocorreu um erro no worker de IA."})
                result_channel = f"results:{job_id}"
                redis_conn.publish(result_channel, error_payload)
                print(f"[!] Mensagem de erro para {job_id} publicada.")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"[*] Cache da GPU limpo após a tarefa (Job ID: {job_id}).")

if __name__ == "__main__":
    try:
        process_tasks()
    except KeyboardInterrupt:
        print("\nWorker interrompido. Limpando recursos...")
