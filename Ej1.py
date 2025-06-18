# Ej1.py
# Ejercicio 1: Extracción y exploración de datos
# Carga datos de 911_times.csv y genera:
#  a) Gráfico del proceso Poisson N(t) para las cuatro franjas
#  b) Histogramas y boxplots de los tiempos entre llamadas filtrando outliers > 3000 s

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('911_times.csv')
    
    # 2. Proceso Poisson (parte a)
    n = min(100_000, len(df))
    plt.figure(figsize=(10, 6))
    for col in ['mañana', 'tarde', 'noche', 'madrugada']:
        times = df[col].iloc[:n].values
        counts = np.arange(1, len(times) + 1)
        plt.step(times, counts, where='post', label=col.capitalize())
    plt.xlabel('Tiempo (s)')
    plt.ylabel('N(t)')
    plt.title('Proceso Poisson empírico (primeras 100k llamadas)')
    plt.legend(title='Franja')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Graficos/Ej1_proceso_poisson.png')
    plt.close()
    
    # 3. Tiempos entre llamadas y filtrado de outliers (parte b)
    diffs = {}
    for col in ['mañana', 'tarde', 'noche', 'madrugada']:
        times = df[col].values
        x = np.diff(times)  # intervalos
        x_filtered = x[x <= 3000]  # filtrar outliers mayores a 3000 s
        diffs[col] = x_filtered
    
    # 4a. Histogramas
    plt.figure(figsize=(12, 8))
    for i, (col, x) in enumerate(diffs.items(), 1):
        plt.subplot(2, 2, i)
        plt.hist(x, bins=50, edgecolor='black')
        plt.title(col.capitalize())
        plt.xlabel('Intervalo (s)')
        plt.ylabel('Frecuencia')
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Graficos/Ej1_histogramas.png')
    plt.close()
    
    # 4b. Boxplots comparativos
    plt.figure(figsize=(8, 6))
    data = [diffs[col] for col in ['mañana', 'tarde', 'noche', 'madrugada']]
    labels = ['Mañana', 'Tarde', 'Noche', 'Madrugada']
    # Usar configuración por defecto de whiskers (1.5*IQR) y mostrar outliers
    plt.boxplot(data, tick_labels=labels)
    plt.title('Boxplots de intervalos entre llamadas\n(filtrado x_k > 3000 s)')
    plt.ylabel('Intervalo (s)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('Graficos/Ej1_boxplots.png')
    plt.close()
    
    # Versión con escala logarítmica para mejor visualización
    plt.figure(figsize=(8, 6))
    plt.boxplot(data, tick_labels=labels)
    plt.title('Boxplots de intervalos entre llamadas (escala log)\n(filtrado x_k > 3000 s)')
    plt.ylabel('Intervalo (s)')
    plt.yscale('log')
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('Graficos/Ej1_boxplots_log.png')
    plt.close()
    
    print("Gráficos generados: Graficos/Ej1_proceso_poisson.png, Graficos/Ej1_histogramas.png, Graficos/Ej1_boxplots.png")

if __name__ == '__main__':
    main()