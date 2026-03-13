# ==========================================================================
#           Arquivo Principal da Aplicação Flask - Epagri.IA
# ==========================================================================
import uuid
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from back.model import initialize_model, search as search_model
from back.GraphState import get_model_from_cache
from huggingface_hub import login

DEFAULT_LLM_MODEL = "google/gemma-2-9b-it"

# --- 1. Inicializacao e Configuracao da Aplicacao ---
app = Flask(__name__, 
            template_folder='app/templates', 
            static_folder='app/static')

app.secret_key = '12345'

# --- 2. DB Temporario --
TEMP_DB = {
    "login": "labdes",
    "password": "bdeswk"
}

# --- INICIALIZACAO DO MODELO ---
print("Iniciando o login no Hugging Face...")
try:
    login("--")
    get_model_from_cache(DEFAULT_LLM_MODEL)
    print("Login no Hugging Face bem-sucedido.")
except Exception as e:
    print(f"Erro no login do Hugging Face: {e}")
initialize_model()

# ==========================================================================
#                           ROTAS DA APLICAÇÃO
# ==========================================================================

# --- 3. Rotas de Autenticacao e Navegacao ---
@app.route('/')
def home():  # Rota de entrada principal (`/`).
    if 'logged_in' in session:
        return redirect(url_for('chat'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Lida com a página e a lógica de login.
    # Aceita dois métodos HTTP:
    # - GET: Simplesmente exibe a página de login (login.html).
    # - POST: É acionado quando o usuario envia o formulario de login.
    error = None
    if request.method == 'POST':
        # Pega os dados enviados pelo formulario HTML.
        login_form = request.form.get('login')
        password_form = request.form.get('password')

        # Valida as credenciais com nosso banco de dados temporario.
        if login_form == TEMP_DB['login'] and password_form == TEMP_DB['password']:
            # Se as credenciais estiverem corretas:
            # 1. Armazena na sessao que o usuario está logado.
            session['logged_in'] = True
            session['username'] = login_form
            # 2. Redireciona para a pagina de chat.
            return redirect(url_for('chat'))
        else:
            # Se as credenciais estiverem erradas, define uma mensagem de erro.
            error = 'Credenciais inválidas. Tente novamente.'

    # Renderiza a pagina de login. Se houver um erro, ele sera passado para o HTML.
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear() # Limpa toda a sessão (logged_in, username, thread_id)
    return redirect(url_for('login'))


# --- 4. Rota Principal da Aplicacao  ---

@app.route('/chat')
def chat():  
    # Exibe a pagina principal do chat.
    if 'logged_in' not in session:
        return redirect(url_for('login'))
    
    if 'thread_id' not in session:
        session['thread_id'] = str(uuid.uuid4())

    return render_template('chat.html')

# --- 5. Rota de pesquisa  ---

@app.route('/search', methods=['POST'])
def search():
    # Recebe uma pergunta e manda para a llm.
    # Retorna os dados em formato JSON.
    
    # Verificação de segurança: só responde se o usuario estiver logado.
    if 'logged_in' not in session:
        return jsonify({"error": "Acesso não autorizado."}), 401
        
    # Pega os dados JSON enviados pelo JavaScript.
    data = request.get_json()
    pergunta = data.get('question')
    thread_id = session.get('thread_id')

    if not pergunta:
        return jsonify({"error": "Nenhuma pergunta foi fornecida."}), 400

    if not thread_id:
        return jsonify({"error": "Sessão inválida. Por favor, faça login novamente."}), 400
    
    try:
        response_data = search_model(pergunta, thread_id)
        
        return jsonify({
            "results": [
                {
                    "descricao": response_data.get("answer", "Erro ao obter resposta."),
                    "sources": response_data.get("sources", [])
                }
            ]
        })
    except Exception as e:
        print(f"ERRO AO PROCESSAR A BUSCA: {e}")
        return jsonify({"error": "Ocorreu um erro interno ao processar sua pergunta. Tente novamente."}), 500


# --- 6. Execução da Aplicacao ---
if __name__ == '__main__':
    # debug=True: Ativa o modo de depuração, que reinicia o servidor automaticamente
   #app.run(debug=True)
   app.run(ssl_context='adhoc')
