import pandas as pd
import os

# Nome do arquivo da etapa anterior
input_file = 'rouge_metrics_table.csv'
output_file = 'metrics_by_question.csv'

# Verificar se o arquivo de entrada existe
if os.path.exists(input_file):
    try:
        # Ler o arquivo CSV gerado anteriormente
        df = pd.read_csv(input_file)

        # Definir as colunas de métricas para calcular a média
        metric_columns = [
            'rouge_1_precision', 'rouge_1_recall', 'rouge_1_f1',
            'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1'
        ]

        # Agrupar por 'question_number' e calcular a média das colunas de métricas
        df_agg_by_question = df.groupby('question_number')[metric_columns].mean()

        # Resetar o índice para que 'question_number' volte a ser uma coluna
        df_agg_by_question = df_agg_by_question.reset_index()

        # Salvar a nova tabela agregada em um novo arquivo CSV
        df_agg_by_question.to_csv(output_file, index=False, float_format='%.6f')

        print(f"Tabela agregada por número da questão salva em: {output_file}")
        print("\nVisão geral das métricas médias por questão:")
        # Usar .to_string() para formatar bem a tabela
        print(df_agg_by_question.to_string(float_format='%.6f'))

    except pd.errors.EmptyDataError:
        print(f"Erro: O arquivo de entrada '{input_file}' está vazio.")
    except Exception as e:
        print(f"Ocorreu um erro ao processar o arquivo: {e}")

else:
    print(f"Erro: Arquivo de entrada '{input_file}' não encontrado.")
    print("Por favor, execute o script anterior primeiro para gerar 'rouge_metrics_table.csv'.")