#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:25:25 2025

@author: aurasofi
"""

#main usando SCRIPTS

import functions_Hurst as func
import pandas as pd
import matplotlib.pyplot as plt


# Leer datos desde archivo CSV
file1 = "/Users/aurasofi/Documents/sp500_cleaned_data.csv"
data = pd.read_csv(file1, index_col=0, parse_dates=True)

file2 = "/Users/aurasofi/Documents/sp500_daily_returns.csv"
returnsfile = pd.read_csv(file2, index_col=0, parse_dates=True)

file3 = "/Users/aurasofi/Documents/sp500_close_stockprices.csv"
close_df = pd.read_csv(file3, index_col=0, parse_dates=True)

file4 = '/Users/aurasofi/Documents/py_Hurst_df.csv'  #csv de python
#file4 = '/Users/aurasofi/Documents/own_Hurst_df.csv'   # csv de own 
#file4 = '/Users/aurasofi/Documents/fd4_Hurst_df.csv'  #cvs de fd4

hurst_df = pd.read_csv(file4, index_col=0, parse_dates=True)




"""
FD4 & OWN & PYTHON  --- uncomment para sacar el exponente de Hurst sin usar el cvs
"""

# # Lista de tickers (columnas del archivo CSV)
# usar = returnsfile.columns


# # DataFrame para almacenar los resultados
# hurst_results = []     #lista vacia  


# # Iterar sobre cada ticker en la lista usar
# for ticker in usar:
   
#     try:

#     #FD4
#         # high_values = high[ticker].dropna().values
#         # low_values = low[ticker].dropna().values
#         # hurst_exponent = func.fd4(high_values, low_values)
        
               
#     #OWN
#         #ts = returnsfile[ticker].dropna()  # Eliminar valores NaN 
#         #hurst_exponent = func.hurst_own(ts)  
          
        
#    #Python
#         ts = returnsfile[ticker].dropna() 
#         hurst_exponent = func.py_hurst(ts, ticker)
    

    
   
#         if 0 <= hurst_exponent <= 1:  # Validar rango esperado
#             hurst_results.append({'Ticker': ticker, 'Hurst Exponent': hurst_exponent})
#         else:
#             print(f"Exponente fuera de rango para {ticker}: {hurst_exponent}")
#     except Exception as e:
#         print(f"Error al calcular exponente para {ticker}: {e}")


## Convertir los resultados a un DataFrame
#hurst_df = pd.DataFrame(hurst_results)


# # Graficar histograma de exponentes H
# plt.hist(hurst_df['Hurst Exponent'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
# plt.axvline(0.5, color='red', linestyle='--', label='Hurst = 0.5')
# plt.xlabel('Exponente de Hurst')
# plt.ylabel('Frecuencia')
# plt.title('Distribución del Exponente de Hurst en el S&P 500')
# plt.legend()
# plt.grid()
# plt.show()



#####################################################################
#Evaluacion de Hurst Exponent y data frame con los tickers ya evaluados

#SIGMA
longitud = returnsfile.index.shape
sigma = func.cal_sigma(longitud)


#Evaluar que esten dentro del intervalo de H y sigma
#selected = func.apply_fil(hurst_df, sigma)


#TEST STATISTIC aplicado a todom los exp. Hurst
test = func.applytest_allHurst(hurst_df,  sigma)
list_test = list(test['Ticker']) 


#Precios de cierre con solo los tickers evaluados del Hurst exp.
usar_selectedH_tickers = list(set(close_df).intersection(set(list_test)))

#close_filtrado = close_df[usar_selectedH_tickers]


#_________________________________________________________________________
# ENTROPY

entropy = func.entropy_filter(returnsfile)
print(entropy) 

list_entropy = list(entropy['Ticker'])
usar_selectedE_tickers = list(set(close_df).intersection(set(list_entropy)))

close_filtrado = close_df[usar_selectedE_tickers]


#_________________________________________________________________________
# ANALISIS DE TEORIA DE PORTAFOLIOS

#MAX SHARPE RATIO AND MIN VOLATILITY
eff_result = func.eff_frontier(close_filtrado)         #Calcula efficient frontier
ef_max = func.max_sharpe(eff_result[0])
ef_min = func.min_volatility(eff_result[1], eff_result[2]) 

print('\n', "Analysis max sharpe ratio:")
analysis_max = func.analysis_port(close_filtrado , ef_max, 10_000)

print('\n', "Analysis min volatility:")
analysis_min = func.analysis_port(close_filtrado , ef_min, 10_000)

#Grafica ambos portafolios
func.plot_portafolios(eff_result[1], eff_result[2], ef_max, ef_min, close_filtrado)


#Black Litterman Model
#func.black_litterman(close_filtrado)





########################## import to EXCEL


# # Guardar el DataFrame en un archivo Excel
#selected.to_excel('/Users/aurasofi/Documents/selec_spy_Hurst.xlsx', index=True, engine='openpyxl')


# imp = test[['Ticker', 'Hurst Exponent']]
# imp.to_excel('/Users/aurasofi/Documents/test_sown_Hurst.xlsx', index=True, engine='openpyxl')

#hurst_df.to_csv('/Users/aurasofi/Documents/own_Hurst_df.csv')