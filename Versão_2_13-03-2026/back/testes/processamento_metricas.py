import json
import pandas as pd

def process_metrics_file(file_name, metric_prefix):
    """
    Lê um arquivo JSON de histórico de execução e extrai as métricas
    individuais (por pergunta).

    Args:
        file_name (str): O nome do arquivo JSON.
        metric_prefix (str): O prefixo da métrica (ex: 'rouge_1' ou 'rouge_l').

    Returns:
        pd.DataFrame: Um DataFrame com os dados processados e nivelados.
    """
    all_scores = []

    try:
        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)

        executions = data.get("executions", [])

        for exec in executions:
            exec_num = exec.get("execution_number")
            model = exec.get("model")

            individual_scores = exec.get("individual_scores", [])

            for score in individual_scores:
                question_num = score.get("question_number")

                # Nomes das chaves de métrica, ex: "rouge_1_recall"
                recall_key = f"{metric_prefix}_recall"
                precision_key = f"{metric_prefix}_precision"
                f1_key = f"{metric_prefix}_f1"

                record = {
                    "execution_number": exec_num,
                    "model": model,
                    "question_number": question_num,
                    precision_key: score.get(precision_key),
                    recall_key: score.get(recall_key),
                    f1_key: score.get(f1_key)
                }
                all_scores.append(record)

    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_name}' não foi encontrado.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{file_name}' não é um JSON válido.")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao processar '{file_name}': {e}")
        return pd.DataFrame()

    return pd.DataFrame(all_scores)

# --- Script Principal ---

# Nomes dos arquivos de entrada
rouge_1_file = 'rouge_1_executions_history.json'
rouge_l_file = 'rouge_l_executions_history.json'

# Processar ambos os arquivos
df_rouge_1 = process_metrics_file(rouge_1_file, 'rouge_1')
df_rouge_l = process_metrics_file(rouge_l_file, 'rouge_l')

# Verificar se os DataFrames não estão vazios antes de mesclar
if not df_rouge_1.empty and not df_rouge_l.empty:
    # Definir as colunas chave para a mesclagem
    merge_keys = ['execution_number', 'model', 'question_number']

    # Mesclar os dois DataFrames usando as chaves
    df_merged = pd.merge(df_rouge_1, df_rouge_l, on=merge_keys)

    # Ordenar os dados para melhor legibilidade
    df_merged = df_merged.sort_values(by=['execution_number', 'model', 'question_number'])

    # Salvar a tabela combinada em um arquivo CSV
    output_csv_file = 'rouge_metrics_table.csv'
    df_merged.to_csv(output_csv_file, index=False)

    print(f"Script executado com sucesso!")
    print(f"Os dados combinados foram salvos em: {output_csv_file}")
    print("\nAmostra dos dados mesclados:")
    print(df_merged.head())

else:
    print("Erro: Um ou ambos os arquivos de entrada não puderam ser processados. Nenhum arquivo de saída foi gerado.")