import os
import warnings
import pandas as pd
from autorank_manual import autorank, create_report, plot_stats
import sys
from io import StringIO
import matplotlib.pyplot as plt

def analyze_all_executions():
    # Ignorar avisos de plotagem e estatística
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", RuntimeWarning)

    input_file = 'rouge_metrics_table.csv'
    output_dir = 'autorank_reports_by_execution'
    METRIC_TO_TEST = 'rouge_1_f1' # Métrica F1 de ROUGE-L

    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo de entrada '{input_file}' não encontrado.")
        return

    # Criar diretório de saída
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Diretório '{output_dir}' criado.")

    try:
        # Carregar todos os dados
        df_all = pd.read_csv(input_file)

        # Obter números de execução únicos
        exec_numbers = sorted(df_all['execution_number'].unique())

        print(f"Iniciando análise com 'autorank' para {len(exec_numbers)} execuções...")

        generated_files = []

        # Loop por cada execução
        for exec_num in exec_numbers:
            print(f"Processando Execução {exec_num}...")

            # 1. Filtrar dados para esta execução
            df_current_exec = df_all[df_all['execution_number'] == exec_num]

            mapping = {"ind_gemma_v2": "Gemma - ST - IQ", "ind_gemma_v4": "Gemma - M - IQ", "seq_gemma_v2": "Gemma - ST - SQ", "seq_gemma_v4": "Gemma - M - SQ",
                   "ind_llama_v2": "Llama - ST - IQ", "ind_llama_v4": "Llama - M - IQ", "seq_llama_v2": "Llama - ST - SQ", "seq_llama_v4": "Llama - M - SQ"}

            df_current_exec['model'] = df_current_exec['model'].map(mapping)
            # 2. Pivotar a tabela (linhas=questões, colunas=modelos)
            df_pivot = df_current_exec.pivot(
                index='question_number',
                columns='model',
                values=METRIC_TO_TEST
            )

            # Remover qualquer linha que tenha NaN
            df_pivot = df_pivot.dropna()

            if df_pivot.empty:
                print(f"  Sem dados completos para a Execução {exec_num}. Pulando.")
                continue

            result = autorank(df_pivot, alpha=0.05, verbose=False, force_mode='nonparametric')

            # 4. Salvar o relatório de texto
            report_filename = os.path.join(output_dir, f'autorank_exec_{exec_num}_{METRIC_TO_TEST}_report.txt')

            # Redirecionar stdout para capturar a saída do create_report
            old_stdout = sys.stdout
            text_trap = StringIO()
            sys.stdout = text_trap

            create_report(result) # Esta função imprime na tela

            sys.stdout = old_stdout # Restaurar stdout

            # Salvar o relatório capturado
            with open(report_filename, 'w') as f:
                f.write(f"RELATÓRIO DE ANÁLISE - EXECUÇÃO {exec_num} - MÉTRICA {METRIC_TO_TEST}\n")
                f.write("="*60 + "\n\n")
                f.write(text_trap.getvalue())
                f.write("\n\n--- DADOS COMPLETOS DO RANKRESULT ---\n")
                f.write(str(result))

            generated_files.append(report_filename)

            # 5. Gerar e salvar o gráfico (se for significante)
            try:
                plot_filename = os.path.join(output_dir, f'autorank_exec_{exec_num}_{METRIC_TO_TEST}_cd_diagram.png')
                # Usamos allow_insignificant=True por segurança, embora já tenhamos verificado
                plot = plot_stats(result, allow_insignificant=True)
                plot.figure.savefig(plot_filename, bbox_inches='tight')
                plt.close(plot.figure) # Fechar a figura para liberar memória
                generated_files.append(plot_filename)

            except ValueError as e:
                # Captura o erro do plot_stats se algo der errado (ex: Wilcoxon)
                print(f"  Não foi possível gerar o gráfico para a Execução {exec_num}: {e}")
            except Exception as e:
                print(f"  Erro inesperado ao gerar gráfico para Execução {exec_num}: {e}")


        print(f"\nProcessamento concluído.")
        print(f"Total de arquivos gerados: {len(generated_files)}")
        print(f"Todos os arquivos foram salvos na pasta: '{output_dir}'")

        # Imprimir uma amostra do que foi criado
        if 'autorank_exec_1_rouge_l_f1_report.txt' in os.listdir(output_dir):
             print(f"\nAmostra do primeiro relatório (autorank_exec_1_rouge_l_f1_report.txt):")
             with open(os.path.join(output_dir, 'autorank_exec_1_rouge_l_f1_report.txt'), 'r') as f:
                print("\n".join(f.readlines()[:15])) # Imprime as primeiras 15 linhas

    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o processamento: {e}")

# --- Executar o script ---
analyze_all_executions()