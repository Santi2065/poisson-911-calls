#!/usr/bin/env python3
# Ej2.py
# Ejercicio 2: Estimación de tasas λ
# Carga datos de 911_times.csv y calcula:
#  a) Estimador de Máxima Verosimilitud (MLE) para λ
#  b) Estimador Bayesiano (media posterior) con prior Gamma(α=1, β=1)
# Reporta los resultados en tabla impresa por consola.

import pandas as pd
import numpy as np

def main():
    # Parámetros de la prior Gamma
    alpha, beta = 1.0, 1.0

    df = pd.read_csv('911_times.csv')
    franjas = ['mañana', 'tarde', 'noche', 'madrugada']
    results = []

    for franja in franjas:
        # Extraer tiempos y calcular intervalos
        times = df[franja].dropna().values
        x = np.diff(times)
        n = len(x)
        sum_x = int(x.sum())

        # MLE: n / Σx
        lambda_mle = n / sum_x

        # Bayesiano (media de la posterior Gamma(α+n, β+Σx))
        lambda_bayes = (alpha + n) / (beta + sum_x)

        results.append({
            'Franja': franja.capitalize(),
            'n (intervalos)': n,
            'Σ x_i (s)': sum_x,
            'λ_MLE (1/s)': lambda_mle,
            'λ_Bayes (media posterior)': lambda_bayes
        })

    # Presentar resultados: Σ x_i como entero, lambdas con 9 decimales
    df_res = pd.DataFrame(results)
    print("\\nEstimaciones de λ por franja:\\n")
    print(
        df_res.to_string(
            index=False,
            float_format='%.9f' # Ajustado a 9 decimales para que se vea bien la diferencia
        )
    )

if __name__ == '__main__':
    main()