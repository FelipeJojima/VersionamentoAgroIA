import pandas as pd
import os

def create_wide_execution_reports():
    """
    Lê o arquivo CSV longo e cria um CSV "largo" separado para cada
    número de execução.

    As linhas são 'question_number', as colunas principais são 'model',
    e as sub-colunas são as métricas.
    """

    # O arquivo CSV de entrada da etapa anterior
    input_file = 'rouge_metrics_table.csv'
    output_dir = 'execution_reports' # Pasta para organizar os arquivos

    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo de entrada '{input_file}' não encontrado.")
        print("Por favor, execute o script anterior primeiro para gerar 'rouge_metrics_table.csv'.")
        return

    # Criar um diretório para salvar os 30 arquivos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório '{output_dir}' criado.")

    try:
        # Carregar o DataFrame longo
        df = pd.read_csv(input_file)

        # Definir a ordem desejada das métricas (sub-colunas)
        metric_order = [
            'rouge_1_precision', 'rouge_1_recall', 'rouge_1_f1',
            'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1'
        ]

        # Obter a lista de modelos na ordem alfabética (colunas principais)
        # O sort é importante para garantir uma ordem consistente em todas as tabelas
        model_order = sorted(df['model'].unique())

        # Obter os números de execução únicos
        exec_numbers = sorted(df['execution_number'].unique())

        output_files = []

        print(f"Iniciando processamento de {len(exec_numbers)} arquivos...")

        # Iterar sobre cada número de execução e criar um arquivo CSV separado
        for exec_num in exec_numbers:
            # 1. Filtrar o DataFrame para a execução atual
            df_filtered = df[df['execution_number'] == exec_num].drop(columns='execution_number')

            # 2. Definir o índice para pivotar
            # Queremos 'question_number' como índice final e 'model' como colunas
            df_indexed = df_filtered.set_index(['question_number', 'model'])

            # 3. Usar unstack() para mover o nível 'model' do índice para as colunas
            # Isso cria um MultiIndex de (métrica, modelo)
            df_wide = df_indexed.unstack(level='model')

            # 4. Trocar os níveis do MultiIndex de coluna para (modelo, métrica)
            df_wide_swapped = df_wide.swaplevel(0, 1, axis=1)

            # 5. Criar o MultiIndex de coluna na ordem correta
            new_column_index = pd.MultiIndex.from_product(
                [model_order, metric_order],
                names=['model', 'metric']
            )

            # 6. Reordenar as colunas para que correspondam à ordem desejada
            df_final = df_wide_swapped.reindex(columns=new_column_index)

            # 7. Ordenar o índice (question_number)
            df_final = df_final.sort_index()

            # Definir o nome do arquivo de saída
            filename = os.path.join(output_dir, f'execution_{exec_num}_wide_metrics.csv')

            # Salvar em CSV, formatando os números
            df_final.to_csv(filename, float_format='%.6f')
            output_files.append(filename)

        print(f"\nArquivos gerados com sucesso: {len(output_files)} arquivos.")
        print(f"Todos os arquivos foram salvos na pasta: '{output_dir}'")

        # Imprimir uma amostra do primeiro arquivo para verificação
        if output_files:
            print(f"\nAmostra das 5 primeiras linhas do arquivo '{output_files[0]}':")
            df_sample = pd.read_csv(output_files[0], header=[0, 1], index_col=0)
            print(df_sample.head().to_string(float_format='%.6f'))

    except pd.errors.EmptyDataError:
        print(f"Erro: O arquivo de entrada '{input_file}' está vazio.")
    except Exception as e:
        print(f"Ocorreu um erro durante o processamento: {e}")

# --- Executar o script ---
if __name__ == "__main__":
    create_wide_execution_reports()