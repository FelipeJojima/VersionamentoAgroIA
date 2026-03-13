import pandas as pd
import os

def create_metric_reports():
    """
    Lê o arquivo CSV longo (rouge_metrics_table.csv) e cria um
    CSV "largo" separado para cada métrica individual, comparando
    todos os modelos.
    """

    # O arquivo CSV de entrada "longo" (fonte da verdade)
    input_file = 'rouge_metrics_table.csv'

    # Pasta para os novos relatórios por métrica
    output_dir = 'metric_reports'

    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo de entrada '{input_file}' não encontrado.")
        print("Por favor, execute o primeiro script novamente para gerá-lo.")
        return

    # Criar um diretório para salvar os 6 arquivos
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório '{output_dir}' criado.")

    try:
        # Carregar o DataFrame longo
        df = pd.read_csv(input_file)

        # Lista das métricas que queremos separar em arquivos
        metrics_to_pivot = [
            'rouge_1_precision', 'rouge_1_recall', 'rouge_1_f1',
            'rouge_l_precision', 'rouge_l_recall', 'rouge_l_f1'
        ]

        output_files = []
        model_order = sorted(df['model'].unique())

        print(f"Processando {input_file} para gerar 6 relatórios por métrica...")

        for metric in metrics_to_pivot:
            # Selecionar apenas as colunas necessárias para este pivô
            df_subset = df[['execution_number', 'question_number', 'model', metric]]

            # Criar a tabela pivô:
            # - Índice (linhas) será uma combinação de execução e questão
            # - Colunas serão os modelos
            # - Valores serão a métrica atual
            df_pivot = df_subset.pivot_table(
                index=['execution_number', 'question_number'],
                columns='model',
                values=metric
            )

            # Ordenar o índice para que fique (1,1), (1,2)... (2,1), (2,2)...
            df_pivot = df_pivot.sort_index()

            # Reordenar as colunas do modelo alfabeticamente para consistência
            df_pivot = df_pivot.reindex(columns=model_order)

            # Definir o nome do arquivo de saída
            filename = os.path.join(output_dir, f'report_{metric}.csv')

            # Salvar em CSV, formatando os números
            df_pivot.to_csv(filename, float_format='%.6f')
            output_files.append(filename)

        print(f"\nArquivos gerados com sucesso: {len(output_files)} arquivos.")
        print(f"Todos os arquivos foram salvos na pasta: '{output_dir}'")

        # Imprimir uma amostra do primeiro arquivo para verificação
        if output_files:
            print(f"\nAmostra das 5 primeiras linhas do arquivo '{output_files[0]}':")
            # Ler de volta para mostrar a aparência
            df_sample = pd.read_csv(output_files[0], index_col=[0, 1])
            print(df_sample.head().to_string(float_format='%.6f'))

    except Exception as e:
        print(f"Ocorreu um erro durante o processamento: {e}")

# --- Executar o script ---
if __name__ == "__main__":
    create_metric_reports()