import sqlite3
import sys

DB_NAME = 'users.db'
conn = None  # Inicializa a variável de conexão como None

try:
    # 1. Tenta conectar ao banco de dados.
    #    O arquivo 'users.db' será criado se não existir.
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print(f"Conexão com '{DB_NAME}' estabelecida com sucesso.")

    # 2. Executa o comando SQL para criar a tabela.
    #    'IF NOT EXISTS' previne um erro caso o script seja executado novamente.
    print("Verificando/Criando a tabela 'users'...")
    cursor.execute('''
    DROP TABLE users
    ''')

    # 3. Confirma (commita) a transação para salvar a criação da tabela.
    conn.commit()
    
    print("Banco de dados e tabela 'users' inicializados com sucesso!")

except sqlite3.OperationalError as e:
    # Captura erros operacionais específicos do SQLite, como:
    # - Falha ao abrir o arquivo do banco (ex: permissão negada).
    # - Sintaxe SQL inválida (improvável aqui, mas bom ter).
    print(f"ERRO OPERACIONAL: Não foi possível configurar o banco de dados.", file=sys.stderr)
    print(f"Detalhe do erro: {e}", file=sys.stderr)
    sys.exit(1) # Encerra o script com um código de erro

except Exception as e:
    # Captura qualquer outra exceção inesperada para diagnóstico.
    print(f"ERRO INESPERADO: Ocorreu um problema durante a inicialização do banco.", file=sys.stderr)
    print(f"Detalhe do erro: {e}", file=sys.stderr)
    sys.exit(1) # Encerra o script com um código de erro

finally:
    # 4. O bloco 'finally' é executado SEMPRE, tenha ocorrido um erro ou não.
    #    Isso garante que a conexão com o banco de dados seja sempre fechada.
    if conn:
        conn.close()
        print(f"Conexão com '{DB_NAME}' foi fechada.")