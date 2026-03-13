# ==========================================================================
#           Monitor de Gerenciamento de Fila - Epagri.IA
# ==========================================================================
# Este script funciona como um painel de controle em tempo real, rodando no
# terminal. Ele se conecta ao Redis e exibe o status das filas de tarefas,
# mostrando quantas tarefas estão aguardando, quantas estão sendo processadas
# e um histórico das últimas tarefas concluídas. É uma ferramenta de
# diagnóstico essencial para entender o que o worker de IA está fazendo.
# ==========================================================================

import redis
import time
import os
import json
from datetime import datetime

# -------------------------------------------------- Configurações --------------------------------------------------
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
RAG_TASKS_QUEUE = "rag_tasks_queue"
TRANSCRIPTION_TASKS_QUEUE = "transcription_tasks_queue"
ALL_TASK_QUEUES = [RAG_TASKS_QUEUE, TRANSCRIPTION_TASKS_QUEUE]
FINISHED_TASKS_LIST = 'finished_tasks'
UPDATE_INTERVAL_SECONDS = 1

# Classe para definir códigos de cores ANSI para o terminal, tornando a saída mais legível
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def clear_screen():
    """Limpa a tela do terminal."""
    os.system('cls' if os.name == 'nt' else 'clear')

def monitor_queues(r):
    """Loop principal que busca e exibe os dados do Redis."""
    print("Iniciando monitoramento... Pressione Ctrl+C para sair.")
    time.sleep(2)  # Pequena pausa inicial

    # Loop infinito que mantém o monitor rodando até ser interrompido.
    while True:
        try:
            clear_screen()
            # --------------------------------------------- 1. Busca de Dados do Redis ---------------------------------------------
            queue_len = 0
            # Itera sobre todas as filas de tarefas para somar o total de itens aguardando
            for q in ALL_TASK_QUEUES:
                queue_len += r.llen(q)

            finished_count = r.llen(FINISHED_TASKS_LIST)
            in_progress_channels = r.pubsub_channels(f'results:*')
            last_finished_raw = r.lrange(FINISHED_TASKS_LIST, 0, 4)

            # ---------------------------------------- 2. Impressão do Painel Formatado ---------------------------------------------
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"{Colors.BOLD}--- Painel de Controle Epagri.IA --- ({now}){Colors.ENDC}")

            # Seção Fila de Espera
            queue_names_display = ', '.join(ALL_TASK_QUEUES)
            color = Colors.OKGREEN if queue_len == 0 else Colors.WARNING

            print(f"\n{Colors.HEADER} Fila de Espera ({queue_names_display}){Colors.ENDC}")
            print(f"  Tarefas aguardando: {color}{queue_len}{Colors.ENDC}")

            # Seção Em Andamento
            in_progress_count = len(in_progress_channels)
            color = Colors.OKGREEN if in_progress_count == 0 else Colors.OKCYAN
            print(f"\n{Colors.HEADER} Em Andamento {Colors.ENDC}")
            print(f"  Tarefas processando: {color}{in_progress_count}{Colors.ENDC}")
            if in_progress_count > 0:
                # Se houver tarefas em andamento, extrai e exibe o ID de cada uma
                for channel in in_progress_channels:
                    job_id = channel.split(':')[-1]
                    print(f"    - Job ID: {job_id}")

            # Seção Finalizados
            print(f"\n{Colors.HEADER} Finalizados {Colors.ENDC}")
            print(f"  Total de tarefas finalizadas: {Colors.OKBLUE}{finished_count}{Colors.ENDC}")
            if last_finished_raw:
                print("  Últimas 5 tarefas:")
                for item_raw in last_finished_raw:
                    try:
                        # Cada item no log é uma string JSON, então precisamos convertê-la de volta
                        item = json.loads(item_raw)
                        job_id = item.get('job_id', 'N/A')
                        question_full = item.get('question', 'N/A')
                        # Trunca a pergunta para que ela não quebre o layout do painel
                        question = (question_full[:37] + '...') if len(question_full) > 40 else question_full
                        finished_time = datetime.fromtimestamp(item.get('finished_at')).strftime('%H:%M:%S')
                        print(f"    - [{finished_time}] {job_id[:8]}...: {question}")
                    except (json.JSONDecodeError, TypeError):
                        print(f"    - Registro malformado: {item_raw}")


            print("\n" + "="*50)
            print(f"{Colors.BOLD}Pressione Ctrl+C para sair{Colors.ENDC}")
            # Pausa pelo intervalo definido antes da próxima atualização
            time.sleep(UPDATE_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("\nMonitoramento encerrado.")
            break
        except redis.exceptions.ConnectionError as e:
            clear_screen()
            print(f"{Colors.FAIL}Erro de conexão com o Redis: {e}{Colors.ENDC}")
            print("Tentando reconectar em 5 segundos...")
            time.sleep(5)

if __name__ == "__main__":
    try:
        redis_conn = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        redis_conn.ping()
        print(f"{Colors.OKGREEN}Conectado ao Redis em {REDIS_HOST}:{REDIS_PORT} com sucesso!{Colors.ENDC}")
    except redis.exceptions.ConnectionError as e:
        print(f"{Colors.FAIL}Não foi possível conectar ao Redis: {e}{Colors.ENDC}")
        exit(1)
    
    monitor_queues(redis_conn)