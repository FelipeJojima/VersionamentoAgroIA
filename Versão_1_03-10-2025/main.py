# ==========================================================================
#           Arquivo Principal da Aplicação Flask - Epagri.IA
# ==========================================================================
# Este arquivo é o ponto de entrada da sua aplicação web. Ele lida com as
# requisições HTTP do navegador, gerencia as sessões dos usuários e se comunica
# com o worker de IA através do Redis.
# ==========================================================================

import os
import uuid
import time
import json
import redis
import traceback 
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from huggingface_hub import login
from langchain_core.messages import AIMessage, HumanMessage 
from langgraph.checkpoint.redis import RedisSaver


# -------------------------- 1. Inicializacao e Configuração da Aplicação  -------------------------- 
# Cria a instância principal da aplicação Flask.
app = Flask(__name__, 
            template_folder='app/templates', # Informa onde estão os arquivos HTML
            static_folder='app/static')      # Informa onde estão os arquivos estáticos (CSS, JS, imagens)

# Chave secreta usada pelo Flask para assinar os cookies de sessão, protegendo contra adulteração
app.secret_key = '12345'

# --- Conexão com o Redis ---
# Estabelece a conexão principal que será usada pela aplicação Flask.
try:
    # decode_responses=True é crucial: garante que os dados vindos do Redis sejam automaticamente convertidos de bytes para strings Python 
    redis_conn = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_conn.ping()  # Testa a conexão para garantir que o servidor está respondendo
    print("Flask App: Conectado ao Redis com sucesso!")
except redis.exceptions.ConnectionError as e:
    print("Flask App: Falha ao conectar ao Redis. Verifique se o serviço está ativo.")

# Nomes das filas no Redis
RAG_TASKS_QUEUE = "rag_tasks_queue"
TRANSCRIPTION_TASKS_QUEUE = "transcription_tasks_queue"

# Cria um diretório temporário para armazenar os arquivos de áudio antes de serem processados
TEMP_AUDIO_FOLDER = '/home/rian/projects/epagri_ia/temp_audio'
os.makedirs(TEMP_AUDIO_FOLDER, exist_ok=True)

# Login na Hugging Face necessário para baixar modelos privados
print("Iniciando o login no Hugging Face...")
try:
    login("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")
    print("Login no Hugging Face bem-sucedido.")
except Exception as e:
    print(f"Erro no login do Hugging Face: {e}")


# ------------------------------ 2. Banco de Dados Temporário de Usuários ----------------------------
TEMP_DB = [{
    "login": "felipe",
    "password": "felipe01"
},
{
    "login": "labdes",
    "password": "bdeswk"
},
{
    "login": "rodrigo",
    "password": "epagritec"
},
{
    "login": "rafael",
    "password": "epagritec"
},
{
    "login": "indianara",
    "password": "epagritec"
}
]

# ==========================================================================
#                           ROTAS DA APLICAÇÃO (ENDPOINTS DA API)
# ==========================================================================
# Rotas definem como a aplicação responde a diferentes URLs


# --------------------------------- 3. Rotas de Autenticacao e Navegacao -----------------------------
@app.route('/')
def home():
    # Rota raiz. Redireciona o usuário para a tela de chat se já estiver logado, caso contrário, para a tela de login.
    if 'logged_in' in session:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Lida com a página e a lógica de login.
    error = None
    if request.method == 'POST':  # Se o usuário enviou o formulário de login
        login_form = request.form.get('login')
        password_form = request.form.get('password')

        # Valida as credenciais com nosso banco de dados temporario.
        for log in TEMP_DB:
            print(log)
            if login_form == log['login'] and password_form == log['password']:
                session['logged_in'] = True
                session['username'] = login_form
                return redirect(url_for('chat'))  # Redireciona para a pagina de chat.
        error = 'Credenciais inválidas. Tente novamente.'
    # Se o método for GET ou se o login falhar, renderiza a página de login.
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    # Limpa todos os dados da sessão do usuário e o redireciona para a página de login.
    session.clear() # Limpa (logged_in, username, thread_id)
    return redirect(url_for('login'))

# ------------------------------------------- 4. Rotas do Chat --------------------------------------
@app.route('/chat')
def chat():  
    # Rota principal que renderiza a interface do chat.
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    # Se o usuário não tiver um ID de conversa na sessão, cria um novo. Isso garante que a primeira visita já inicie uma conversa.
    if 'thread_id' not in session:
        session['thread_id'] = str(uuid.uuid4())

    return render_template('chat.html')

@app.route('/new_chat', methods=['POST'])
def new_chat():
    # API para criar uma nova conversa.
    if 'logged_in' not in session:
        return jsonify({"error": "Acesso não autorizado."}), 401
    
    # Substitui o ID de conversa antigo na sessão por um novo.
    session['thread_id'] = str(uuid.uuid4())
    print(f"Nova conversa iniciada. Novo Thread ID: {session['thread_id']}")
    return jsonify({"success": True, "thread_id": session['thread_id']})

@app.route('/get_chats', methods=['GET'])
def get_chats():
    # API para buscar a lista de conversas salvas do usuário para a barra lateral.
    if 'logged_in' not in session or 'username' not in session:
        return jsonify({"error": "Acesso não autorizado."}), 401

    username = session['username']
    chat_ids_key = f"user_chats:{username}"  # Chave da lista de IDs no Redis.
    
   # LRANGE busca todos os elementos da lista.
    thread_ids = redis_conn.lrange(chat_ids_key, 0, -1)
    
    chats = []
    # Itera sobre cada ID para buscar seu título correspondente.
    for thread_id in thread_ids:
        thread_meta_key = f"thread_meta:{thread_id}"
        title = redis_conn.hget(thread_meta_key, "title")

        if title:
            chats.append({
                "id": thread_id, 
                "title": title  
            })
            
    return jsonify(chats)  # Retorna a lista de conversas como JSON.

@app.route('/load_chat', methods=['POST'])
def load_chat():
    # API para trocar a conversa ativa na sessão do usuário
    if 'logged_in' not in session:
        return jsonify({"error": "Acesso não autorizado."}), 401
        
    data = request.get_json()
    thread_id = data.get('thread_id')
    
    if not thread_id:
        return jsonify({"error": "ID da conversa não fornecido."}), 400
    
    # Atualiza a sessão para que a próxima mensagem enviada vá para esta conversa
    session['thread_id'] = thread_id
    print(f"Carregada conversa com Thread ID: {thread_id}")
    return jsonify({"success": True})

# ---------------------------------- 5. Rota Principal de Processamentos de IAs -------------------------
@app.route('/search', methods=['POST'])
def search():
    # Endpoint principal que recebe a pergunta do usuário e gerencia a comunicação com o worker.
    if 'logged_in' not in session:
        return jsonify({"error": "Acesso não autorizado."}), 401
        
    data = request.get_json()
    pergunta = data.get('question')
    thread_id = session.get('thread_id')
    username = session.get('username')

    if not pergunta or not thread_id or not username:
        return jsonify({"error": "Dados inválidos."}), 400
    
    # Lógica para salvar o título da conversa na primeira mensagem.
    try:
        user_chats_key = f"user_chats:{username}"
        thread_meta_key = f"thread_meta:{thread_id}"
        
        # HGET verifica se o campo "title" já existe.
        if not redis_conn.hget(thread_meta_key, "title"):
            print(f"Primeira mensagem na thread {thread_id}. Salvando conversa...")
            chat_title = (pergunta[:100] + '...') if len(pergunta) > 100 else pergunta
            redis_conn.hset(thread_meta_key, "title", chat_title)  # Salva o título.
            redis_conn.lpush(user_chats_key, thread_id)  # Adiciona o ID à lista do usuário.
    except Exception as e:
        print(f"ERRO ao salvar metadados da conversa no Redis: {e}")

    pubsub = None
    try:
        job_id = str(uuid.uuid4())  # ID único para esta requisição específica.
        task_data = { "pergunta": pergunta, "thread_id": thread_id, "job_id": job_id }

        # Usa o padrão Pub/Sub do Redis para comunicação assíncrona.
        pubsub = redis_conn.pubsub()
        result_channel = f"results:{job_id}"  # Canal único para ouvir a resposta.
        pubsub.subscribe(result_channel)
        
        # LPUSH envia a tarefa para a fila que o worker está ouvindo.
        redis_conn.lpush(RAG_TASKS_QUEUE, json.dumps(task_data))

        response = None
        start_time = time.time()
        timeout_seconds = 10 * 60  # espera até 10 minutos

        # Loop de espera pela resposta do worker
        while time.time() - start_time < timeout_seconds:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
            if message:
                response = json.loads(message['data'])
                break   # Sai do loop assim que a resposta chega.
        
        if response is None:
             return jsonify({"error": "O serviço de IA está demorando para responder."}), 504

        # Lógica para salvar o histórico da conversa (pergunta e resposta) no Redis.
        if response and "answer" in response:
            try:
                history_key = f"chat_history:{thread_id}"
                
                user_message = json.dumps({"type": "user", "content": pergunta})
                redis_conn.rpush(history_key, user_message)

                bot_message_data = {
                    "type": "bot",
                    "content": response.get("answer"),
                    "sources": response.get("sources", [])
                }
                
                bot_message = json.dumps(bot_message_data)
                redis_conn.rpush(history_key, bot_message)
                
                print(f"Pergunta e Resposta (com fontes) salvas no histórico do Redis para thread {thread_id}")

            except Exception as e:
                print(f"ERRO ao salvar histórico da conversa no Redis: {e}")

        # Retorna a resposta da IA para o frontend.
        return jsonify({
            "results": [{"descricao": response.get("answer", ""), "sources": response.get("sources", [])}]
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Ocorreu um erro interno ao processar sua pergunta."}), 500
    finally:
        # Garante que a inscrição no canal seja encerrada, mesmo se ocorrer um erro.
        if pubsub:
            pubsub.unsubscribe()
            pubsub.close()

@app.route('/transcribe', methods=["POST"])
def transcribe():
    # Lida com o upload de áudio e envia para a fila de transcrição.
    if 'logged_in' not in session:
        return jsonify({"error": "Acesso não autorizado."}), 401
    
    if 'audio_data' not in request.files:
        # <-- CORREÇÃO AQUI (padronizando a chave de erro)
        return jsonify({"error": "Nenhum arquivo de áudio enviado."}), 400
    
    audio_file = request.files['audio_data']
    filename = f"{uuid.uuid4()}.webm"
    filepath = os.path.join(TEMP_AUDIO_FOLDER, filename)
    audio_file.save(filepath)

    job_id = str(uuid.uuid4())
    task_data = { "job_id": job_id, "filepath": filepath }

    pubsub = None
    try:
        pubsub = redis_conn.pubsub()
        result_channel = f"results:{job_id}"
        pubsub.subscribe(result_channel)

        redis_conn.lpush(TRANSCRIPTION_TASKS_QUEUE, json.dumps(task_data))

        response = None
        start_time = time.time()
        timeout_seconds = 3 * 60

        while time.time() - start_time < timeout_seconds:
            message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1)
            if message:
                if isinstance(message['data'], bytes):
                    response_data_json = message['data'].decode('utf-8')
                else:
                    response_data_json = message['data']
                response = json.loads(response_data_json)
                break
        
        if response is None:
            return jsonify({"error": "Serviço de transcrição demorou para responder."}), 504
                    
        return jsonify(response)
    finally:
        if pubsub:
            pubsub.unsubscribe()
            pubsub.close()

# ---------------------------------- 6. Rotas de Gerenciamento de Histórico -----------------------------

@app.route('/get_chat_history/<string:thread_id>', methods=['GET'])
def get_chat_history(thread_id):
    # API que busca nosso histórico de mensagens simples do Redis
    if 'logged_in' not in session:
        return jsonify({"error": "Acesso não autorizado."}), 401

    try:
        history_key = f"chat_history:{thread_id}"
        messages_json = redis_conn.lrange(history_key, 0, -1)
        
        # Converte a lista de strings JSON em uma lista de objetos Python
        formatted_history = [json.loads(msg) for msg in messages_json]
        
        return jsonify(formatted_history)

    except Exception as e:
        print(f"ERRO ao buscar histórico do Redis para thread {thread_id}: {e}")
        traceback.print_exc() 
        return jsonify({"error": "Não foi possível carregar o histórico da conversa."}), 500

@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    # API para apagar permanentemente todos os dados de uma conversa
    if 'logged_in' not in session or 'username' not in session:
        return jsonify({"error": "Acesso não autorizado."}), 401

    data = request.get_json()
    thread_id = data.get('thread_id')
    username = session['username']

    if not thread_id:
        return jsonify({"error": "ID da conversa não fornecido."}), 400

    try:
        user_chats_key = f"user_chats:{username}"
        thread_meta_key = f"thread_meta:{thread_id}"
        chat_history_key = f"chat_history:{thread_id}"

        # LREM remove o elemento da lista. É uma verificação de segurança
        if redis_conn.lrem(user_chats_key, 1, thread_id) == 0:
            return jsonify({"error": "Conversa não encontrada ou permissão negada."}), 404

        # Apaga os dados associados
        redis_conn.delete(thread_meta_key)  # Apaga o título
        redis_conn.delete(chat_history_key)  # Apaga as mensagens
        
        # Apaga o histórico do LangGraph (se houver algum resquício)
        keys_to_delete = redis_conn.keys(f"thread_state:{thread_id}:*")
        if keys_to_delete:
            redis_conn.delete(*keys_to_delete)

        print(f"Conversa {thread_id} apagada com sucesso para o usuário {username}.")
        return jsonify({"success": True})

    except Exception as e:
        print(f"ERRO ao apagar a conversa {thread_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": "Ocorreu um erro interno ao apagar a conversa."}), 500
    
# ---------------------------------------- 7. Execução da Aplicacao -------------------------------------
if __name__ == '__main__':
   # debug=True: Ativa o modo de depuração, que reinicia o servidor automaticamente
   # app.run(ssl_context='adhoc')
   app.run()
