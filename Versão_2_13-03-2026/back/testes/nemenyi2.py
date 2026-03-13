import pandas as pd
import os
import warnings
import sys
from io import StringIO
from autorank_manual import autorank, create_report, plot_stats
import matplotlib.pyplot as plt

def run_average_analysis(df_all, metric_to_test, output_dir):
    """
    Executa a análise de autorank para uma métrica específica com base na
    média das 30 execuções.
    """
    print(f"\n--- Iniciando Análise para: {metric_to_test} ---")
    print(df_all)

    try:
        # 1. Calcular a média do F1-score para cada (modelo, questão)
        print(f"  Calculando a média de '{metric_to_test}'...")
        df_avg = df_all.groupby(['model', 'question_number'])[metric_to_test].mean().reset_index()
        # print(df_avg)
        # print(type(df_avg))
        
        mapping = {"ind_gemma_v2": "Gemma - ST - IQ", "ind_gemma_v4": "Gemma - M - IQ", "seq_gemma_v2": "Gemma - ST - SQ", "seq_gemma_v4": "Gemma - M - SQ",
                   "ind_llama_v2": "Llama - ST - IQ", "ind_llama_v4": "Llama - M - IQ", "seq_llama_v2": "Llama - ST - SQ", "seq_llama_v4": "Llama - M - SQ"}

        df_avg['model'] = df_avg['model'].map(mapping)
        # print(df_avg)
        # 2. Pivotar a tabela de médias
        df_pivot_avg = df_avg.pivot(
            index='question_number',
            columns='model',
            values=metric_to_test
        )

        # 3. Verificar se há NaNs
        df_pivot_avg = df_pivot_avg.dropna()
        if df_pivot_avg.empty:
            print(f"  A tabela pivotada para '{metric_to_test}' está vazia. Pulando.")
            return
        print(df_pivot_avg)
        print("  Executando autorank (Friedman + Nemenyi) sobre as médias...")
        # 4. Executar o Autorank
        result = autorank(
            df_pivot_avg,
            alpha=0.05,
            verbose=False,
            order='ascending',
            force_mode='nonparametric' # Força Friedman + Nemenyi
        )

        # 5. Geração de Relatório de Texto
        report_filename = os.path.join(output_dir, f'autorank_average_report_{metric_to_test}.txt')
        print(f"  Salvando relatório de texto em: {report_filename}")

        old_stdout = sys.stdout
        text_trap = StringIO()
        sys.stdout = text_trap
        create_report(result) # Captura a saída
        sys.stdout = old_stdout # Restaura a saída

        with open(report_filename, 'w') as f:
            f.write(f"RELATÓRIO DE ANÁLISE - MÉDIA DAS 30 EXECUÇÕES - MÉTRICA {metric_to_test}\n")
            f.write("="*70 + "\n\n")
            f.write(text_trap.getvalue())
            f.write("\n\n--- DADOS COMPLETOS DO RANKRESULT ---\n")
            f.write(str(result))

        plot_filename = os.path.join(output_dir, f'autorank_average_cd_diagram_{metric_to_test}.png')
        print(f"  P-valor (p={result.pvalue:.4f}) é significante. Gerando diagrama de CD em: {plot_filename}")
        try:
            plot = plot_stats(result, allow_insignificant=True)
            plot.figure.savefig(plot_filename, bbox_inches='tight')
            plt.close(plot.figure) # Fecha a figura
        except Exception as e:
            print(f"  Erro inesperado ao gerar gráfico: {e}")


        print(f"--- Análise para {metric_to_test} concluída. ---")

    except KeyError:
        print(f"Erro: A métrica '{metric_to_test}' não foi encontrada no arquivo CSV.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante a análise de '{metric_to_test}': {e}")


def analyze_all_average_metrics():
    """
    Script principal para analisar as médias de ROUGE-1 F1 e ROUGE-L F1.
    """
    # Ignorar avisos
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)

    input_file = 'rouge_metrics_table.csv'
    output_dir = 'autorank_average_reports' # Pasta para salvar ambos os relatórios

    # Lista de métricas para testar
    METRICS_TO_TEST = ['rouge_1_f1', 'rouge_l_f1']

    # --- Verificação de Arquivos e Pastas ---
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo de entrada '{input_file}' não encontrado.")
        print("Por favor, execute os scripts anteriores para gerar este arquivo.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório '{output_dir}' criado.")

    try:
        # Carregar os dados apenas uma vez
        print(f"Carregando dados de '{input_file}'...")
        df_all = pd.read_csv(input_file)

        # Verificar se as colunas necessárias existem
        if not all(metric in df_all.columns for metric in METRICS_TO_TEST):
            print("Erro: O arquivo CSV não contém as colunas necessárias ('rouge_1_f1', 'rouge_l_f1').")
            print(f"Colunas encontradas: {df_all.columns.tolist()}")
            return

        # Loop para executar a análise para cada métrica
        for metric in METRICS_TO_TEST:
            run_average_analysis(df_all, metric, output_dir)

        print(f"\nProcessamento concluído.")
        print(f"Todos os relatórios foram salvos na pasta: '{output_dir}'")

    except Exception as e:
        print(f"Ocorreu um erro inesperado no script principal: {e}")

# --- Executar o script ---
analyze_all_average_metrics()