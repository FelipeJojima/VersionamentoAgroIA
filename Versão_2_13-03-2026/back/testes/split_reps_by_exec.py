import pandas as pd
import os

def split_reports_by_execution():
    """
    Lê os 6 relatórios de métricas e os divide por número de execução,
    gerando 180 arquivos CSV.
    """

    # Diretório de entrada (onde estão os 6 arquivos)
    input_dir = 'metric_reports'

    # Novo diretório de saída para os 180 arquivos
    output_dir = 'final_reports_by_metric_and_exec'

    # Lista dos 6 arquivos de métricas que criamos na etapa anterior
    metric_files = [
        'report_rouge_1_precision.csv', 'report_rouge_1_recall.csv', 'report_rouge_1_f1.csv',
        'report_rouge_l_precision.csv', 'report_rouge_l_recall.csv', 'report_rouge_l_f1.csv'
    ]

    # Verificar se o diretório de entrada existe
    if not os.path.exists(input_dir):
        print(f"Erro: Diretório de entrada '{input_dir}' não encontrado.")
        print("Por favor, execute o script anterior primeiro.")
        return

    # Criar o diretório de saída para os 180 arquivos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório '{output_dir}' criado.")

    total_files_generated = 0

    print(f"Iniciando a divisão de {len(metric_files)} arquivos de métricas...")

    # 1. Loop principal: Itera sobre cada um dos 6 arquivos de métricas
    for metric_file_name in metric_files:
        input_path = os.path.join(input_dir, metric_file_name)

        if not os.path.exists(input_path):
            print(f"Aviso: Arquivo de entrada '{input_path}' não encontrado. Pulando.")
            continue

        try:
            # 2. Carrega o arquivo de métrica.
            # O índice é composto por 'execution_number' e 'question_number'
            df_metric = pd.read_csv(input_path, index_col=['execution_number', 'question_number'])

            # 3. Obtém a lista de números de execução únicos (ex: 1, 2, ..., 30)
            exec_numbers = df_metric.index.get_level_values('execution_number').unique()

            # 4. Loop aninhado: Itera por cada número de execução
            for exec_num in exec_numbers:

                # 5. Seleciona os dados apenas para essa execução.
                # O .loc[exec_num] retorna um DataFrame indexado por 'question_number'
                df_exec_slice = df_metric.loc[exec_num]

                # 6. Define o nome do arquivo de saída
                # ex: 'rouge_1_precision_exec_1.csv'
                base_name = metric_file_name.replace('report_', '').replace('.csv', '')
                output_filename = f"{base_name}_exec_{exec_num}.csv"
                output_path = os.path.join(output_dir, output_filename)

                # 7. Salva o CSV fatiado
                df_exec_slice.to_csv(output_path, float_format='%.6f')
                total_files_generated += 1

        except Exception as e:
            print(f"Erro ao processar o arquivo '{input_path}': {e}")

    print(f"\nProcessamento concluído.")
    print(f"Total de arquivos gerados: {total_files_generated}")
    print(f"Todos os arquivos foram salvos na pasta: '{output_dir}'")

    # Imprimir uma amostra do que foi criado
    if total_files_generated > 0:
        sample_file_path = os.path.join(output_dir, 'rouge_1_precision_exec_1.csv')
        if os.path.exists(sample_file_path):
            print(f"\nAmostra do primeiro arquivo gerado ({sample_file_path}):")
            df_sample = pd.read_csv(sample_file_path, index_col=0)
            print(df_sample.head().to_string(float_format='%.6f'))
        else:
             print("\nArquivo de amostra não encontrado para exibição.")

# --- Executar o script ---
if __name__ == "__main__":
    split_reports_by_execution()