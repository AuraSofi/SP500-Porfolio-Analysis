#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:58:45 2025

@author: aurasofi
"""

#FUNCIONES

import numpy as np
from scipy.stats import linregress, entropy, norm
from hurst import compute_Hc  
import math
import pandas as pd


from pypfopt import EfficientFrontier, BlackLittermanModel, objective_functions
from pypfopt import risk_models, expected_returns, plotting
import matplotlib.pyplot as plt

from copy import deepcopy

import seaborn as sns

from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


"""
FD4 Calcula el exponente fractal (fd4) basado en series de máximos y mínimos.
"""
# Función mejorada fd4
def fd4(high, low, q=0.01, nmin=0, nmax=-2):

    def k(high, low, q, tau):
        aux = []
        for i in range(len(high) - tau + 1):
            try:
                maximo = np.log(max(high[i:i + tau]))
                minimo = np.log(min(low[i:i + tau]))
                aux.append((maximo - minimo) ** q)
            except ValueError:
                aux.append(0)  # Manejar posibles errores en logaritmos
        return np.sum(aux) / (len(high) - tau + 1.0)

    nmax = int(np.log2(len(high))) + nmax
    rango = [2 ** n for n in range(nmin, nmax)]
    kq = [k(high, low, q, tau) for tau in rango]
    b1 = np.log(rango)
    b2 = np.log(kq)
    lr = linregress(b1, b2)

    return lr.slope / q


"""
HURST PYTHON
"""


def py_hurst(ts, ticker, longitud_minima=100):

    # Verificar que la serie no esté vacía y tenga la longitud mínima
    if len(ts) >= longitud_minima:
        try:
            # Calcular el exponente de Hurst
            H, c, _ = compute_Hc(ts.values, kind='change')  # `kind='change'` para rendimientos diarios
            return H
        except Exception as e:
            print(f"Error para {ticker}: {e}")
            return {"Ticker": ticker, "Hurst Exponent": np.nan}
    else:
        print(f"{ticker}: Serie demasiado corta ({len(ts)} valores)")
        return {"Ticker": ticker, "Hurst Exponent": np.nan}



"""
HURST OWN
"""

def hurst_own(ts):
    def div_in_subsamples(ts):       # Dividir la serie en subseries
        N = len(ts)
        max_lag = 512
        subsamples_R = {}
    
        # Iniciar en una particion de 4 y dividir hasta max_lag
        observations = 4
        while observations <= max_lag:
            # Dividir la serie en subseries de la longitud actual(observations)
            subsamples = [ts[i:i + observations] for i in range(0, N, observations) if len(ts[i:i + observations]) == observations]
             
            # Guardar las subseries para la longitud actual
            subsamples_R[observations] = subsamples
            
            # Pasar a la siguiente longitud (multiplicando por 2)
            observations *= 2
        return subsamples_R
    
    # Dividir la serie de rendimientos diarios en subseries
    ts = div_in_subsamples(ts)
    
    # tabla para almacenar los resultados
    group_rescaled_data = {}
    
    # Iterar sobre cada tamaño de subserie
    for observations, subsamples in ts.items():
        rescaled_data = []
        for i, obser in enumerate(subsamples):
            mean = np.mean(obser)            # Calcular la media de la subserie
            adjusted_returns = obser - mean    # Restar la media a cada valor de la subserie
            cumulative_sum = np.cumsum(adjusted_returns) # Calcular la suma acumulativa de los retornos ajustados
            max_range = np.max(cumulative_sum) - np.min(cumulative_sum)    #Rango maximo de csum
           
            standard_d = obser.std()
            if standard_d == 0:  # Evitar división por cero
                continue  # O asignar un valor predeterminado como 0 o np.nan
           
            rescaled_range = max_range / standard_d 
            rescaled_data.append(rescaled_range)
     
        # Solo calcular la media si hay datos válidos
        if rescaled_data:
            group_rescaled_data[observations] = np.mean(rescaled_data)
    
    # Si no hay datos válidos, retornar np.nan
    if not group_rescaled_data:
        return np.nan    
              
    
    log_rs = {obs: np.log(avg) for obs, avg in group_rescaled_data.items()}
    log_n = {obs: np.log(obs) for obs in group_rescaled_data.keys()}
    
    
    # Convertir los datos a listas
    x = list(log_n.values())  # Log observaciones
    y = list(log_rs.values())  # Log de los promedios rs
    
    # Ajustar una línea recta y calcular la pendiente
    slope, intercept = np.polyfit(x, y, 1)  # Ajuste lineal de grado 1
    return slope


#SIGMA

def cal_sigma(size):
    N = int(size[0])
    e = math.e
    result = 1 / (e * N **(1/3))
    print('Sigma:',result)
    return result



#TEST STATISTIC 

def test_statistic (H, sigma_result):
    H_0 = 0.5 
    t_value = (H - H_0 )/ sigma_result
    p_value = 2 * (1 - norm.cdf(abs(t_value))) #abs - valor absoluto t

    return t_value, p_value

#si p-valor < alpha (0.05) entonces la empresa la integramos al portafolio,

def applytest_allHurst (df, sigma_result):
    alpha=0.05
    df[['t_statistic', 'p_value']] = df['Hurst Exponent'].apply(
        lambda H: pd.Series(test_statistic(H, sigma_result)))
    filtered_df = df[df['p_value'] < alpha].reset_index(drop=True)
    
    if not filtered_df.empty:
        print("Rechaza la hipótesis nula, H difiere significativamente de 0.5")
    else:
        print("No se rechaza la hipótesis nula, H no difiere significativamente de 0.5")

    # Agregar interpretación básica al DataFrame de resultados
    filtered_df[" Basic Interpretation"] = pd.Series(df["Hurst Exponent"].apply(
        lambda x: "Mean-reverting" if x < 0.5 else "Random walk" if x == 0.5 else "Persistent trend"))
    
    return filtered_df




#FILTRO: exp. Hurst + or - sigma

def filtro_sig (sigma_result):
    H = 0.5
    H_sum = H + sigma_result
    H_rest = H - sigma_result
    return H_sum, H_rest

    
    
def apply_fil(df, sigma_result):
    H_sum, H_rest = filtro_sig(sigma_result)
    df_drop = df[(df['Hurst Exponent'] >= H_rest) & (df['Hurst Exponent'] <= H_sum)]
    df_filtered = df[(df['Hurst Exponent'] < H_rest) | (df['Hurst Exponent'] > H_sum)]
    return df_filtered 



def eff_frontier(df):
    mu = expected_returns.mean_historical_return(df)    
    S = risk_models.sample_cov(df)  
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    return ef, mu, S


def max_sharpe(ef):
    ef.max_sharpe()
    ef.deepcopy()
    return ef 


def min_volatility(mu, S):     
    ef = EfficientFrontier(mu, S)
    #ef.add_objective(objective_functions.L2_reg)  # Agregar regularización L2/ penalización de Ridge
    ef.min_volatility()  # Optimizar para minimizar la volatilidad
    return ef




def analysis_port(df, ef, total_portfolio_value):
    cleaned_weights = ef.clean_weights()   #limpia los datos 
    ef.portfolio_performance(verbose=True)
    
    latest_prices = get_latest_prices(df)

    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value)
    allocation, leftover = da.greedy_portfolio()
    print("Discrete allocation:", allocation)
    print("Funds remaining: ${:.2f}".format(leftover))
    
    # Calcular el capital asignado a cada empresa
    capital_allocation = {ticker: shares * latest_prices[ticker] for ticker, shares in allocation.items()}
    
    print("\nCapital asignado a cada empresa:")
    for ticker, capital in capital_allocation.items():
        capital = print(f"{ticker}: ${capital:.2f}")
    return capital
    



def plot_portafolios(mu, S, eff_max, eff_min, df):
    eff_plot = EfficientFrontier(mu, S, weight_bounds=(0,1))
    eff_plot_copy = deepcopy(eff_plot) 
    
    #graficar eff sin optimizar
    fig, ax = plt.subplots()
    plotting.plot_efficient_frontier(eff_plot, ax=ax, aplha=0.7, show_assets=False)
    for line in ax.get_lines():
        line.set_linewidth(2)  # Hace la línea más gruesa
        line.set_linestyle("-.")  # Línea punteada
        
        
        # Obtener los puntos de los activos individuales
    asset_returns = eff_plot.expected_returns
    asset_volatilities = np.sqrt(np.diag(eff_plot.cov_matrix))

    # Graficar los activos individuales con un color específico y agregar los tickers
    ax.scatter(asset_volatilities, asset_returns, marker="o", color="purple", s=75, label="Individual Assets", alpha=0.5)
    
    
    #     # Agregar los tickers (nombres de los activos) junto a los puntos
    # for i, ticker in enumerate(df):
    #     ax.text(asset_volatilities[i], asset_returns[i], ticker, fontsize=6, ha="right", va="bottom")

    

    #Graficar min vol
    ret, vol, sharpe = eff_min.portfolio_performance()    
    # Graficar el portafolio de mínima volatilidad como un punto
    ax.scatter(vol, ret, marker="^", color="navy", s=200, label="Min Volatility") 
    
    
    # Find the tangency portfolio  - graficar max Sharpe ratio 
    ret_tangent, std_tangent, _ = eff_max.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="^", s=200, color="firebrick", label="Max Sharpe")
    

    # Agregar valores a la grafica
    ax.annotate(f"({std_tangent:.2f}, {ret_tangent:.2f})", 
                (std_tangent, ret_tangent), 
                textcoords="offset points", 
                xytext=(10,10), 
                ha='center', fontsize=10, c='firebrick', fontweight='bold')
    
    ax.annotate(f"({vol:.2f}, {ret:.2f})", 
                (vol, ret), 
                textcoords="offset points", 
                xytext=(1,7), 
                ha='center', fontsize=10, c='navy', fontweight='bold')
    

    # Configuración de la gráfica
    ax.set_title("Efficient Frontier, portfolios and individual assets")
    ax.set_xlabel("Volatility (Risk)")
    ax.set_ylabel("Expected Return")
    ax.legend()
    plt.show() 
    

    
def black_litterman(df):
    S = risk_models.sample_cov(df)
    viewdict = {"AAPL": 0.20, "BBY": -0.30, "NVDA": .35, "SBUX": 0.05, "T": 0.12}    #Creencias del inversionista
    bl = BlackLittermanModel(S, pi="equal", absolute_views=viewdict, omega="default")
    rets = bl.bl_returns()
    ef = EfficientFrontier(rets, S)
    ef.max_sharpe()
    print("Retornos esperados ajustados:")
    print(rets)
    ef.clean_weights()  # Limpia los pesos para que sean más legibles
    perf = ef.portfolio_performance(verbose=True)  # Muestra las métricas
    return perf






    """
    Entropia de Shanon
    Filtrar mercados basados en la entropía de Shannon de sus retornos.

    Parámetros:
    - df: daily returns de cada empresa.
    - bins: Número de bins para calcular la distribución de probabilidad.

    Retorna:
    - dict con listas de empresas en el primer y cuarto cuartil de entropía.
    """

def entropy_filter(df, bins=30):
    entropies = {}

    for ticker in df.columns:                   # Obtener la distrib. de probabilidad estimada
        hist, bin_0edges = np.histogram(df[ticker].dropna(), bins=bins, density=True)
        hist = hist / hist.sum()  # Normalizacion a distr. de proba
        
        # Calcular entropia de Shannon
        entropies[ticker] = entropy(hist, base=2)  # Base 2 para entropía en bits

    # Convertir a DataFrame
    entropy_df = pd.DataFrame.from_dict(entropies, orient='index', columns=['Entropy'])
    entropy_df = entropy_df.rename_axis('Ticker').reset_index()
    
    # Calcular cuartiles
    q1 = entropy_df['Entropy'].quantile(0.25)
    q4 = entropy_df['Entropy'].quantile(0.75)
 

    # Poner etiquetas 
    entropy_df['Analysis'] = 'None'  # Valor por defecto
    entropy_df.loc[entropy_df['Entropy'] <= q1, 'Analysis'] = 'Low Entropy'
    entropy_df.loc[entropy_df['Entropy'] >= q4, 'Analysis'] = 'High Entropy'
    
    #Graficar histograma
    plt.figure(figsize=(10, 5))
    sns.histplot(entropy_df['Entropy'], bins=bins, kde=True, color="blue")    #kde - agrega una curva de densidad
    plt.axvline(entropy_df['Entropy'].quantile(0.25), color='red', linestyle='dashed', label='Q1 (Low Entropy)')
    plt.axvline(entropy_df['Entropy'].quantile(0.75), color='green', linestyle='dashed', label='Q4 (High Entropy)')
    
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.title("Distribution of Stock Entropies")
    plt.legend()
    plt.show()
    
    # Filtrar y devolver solo las filas con "Low Entropy"
    loworhigh_entropy_df = entropy_df[entropy_df['Analysis'] == 'Low Entropy']

    return loworhigh_entropy_df





    






