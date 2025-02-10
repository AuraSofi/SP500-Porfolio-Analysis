#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 19:59:55 2025

@author: aurasofi
"""

import yfinance as yf
import pandas as pd



#Imprimir los tickers del S&P500
tickers = pd.read_html(
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies' )[0]
print (tickers.head())


#Obtener los datos de cada ticker de yf
data = yf.download(tickers.Symbol.to_list(), start="2012-01-01", end="2024-01-01", auto_adjust=True)
high = data['High']
low = data['Low']
close = data['Close']

#auto_adjust = True ajusta automaticamente los precios históricos de las acciones para splits y dividendos

# Filtro: Empresas con menos del 10% de datos NaN
usar = []
eliminar = []

for ticker in close.columns:
    total_values = len(close[ticker])
    nan_count = close[ticker].isnull().sum()
    nan_percentage = nan_count / total_values

    if nan_percentage < 0.10:
        usar.append(ticker)
    else:
        eliminar.append(ticker)

# Imputar datos - close, high, low
for columna in usar:
    if close[columna].isnull().all():
        close[columna].fillna(0, inplace=True)
        high[columna].fillna(0, inplace=True)
        low[columna].fillna(0, inplace=True)
    else:
        close[columna].ffill(inplace=True)
        close[columna].bfill(inplace=True)
        high[columna].ffill(inplace=True)
        high[columna].bfill(inplace=True)
        low[columna].ffill(inplace=True)
        low[columna].bfill(inplace=True)

# Eliminar columnas no usadas
close = close.drop(columns=eliminar)
high = high.drop(columns=eliminar)
low = low.drop(columns=eliminar)


# Calcular los rendimientos diarios
daily_returns = close.pct_change() 



# Guardar los datos limpios y procesados en un archivo CSV
#data.to_csv("/Users/aurasofi/Documents/sp500_cleaned_data.csv")
#daily_returns.to_csv("/Users/aurasofi/Documents/sp500_daily_returns.csv")
#close.to_csv("/Users/aurasofi/Documents/sp500_close_stockprices.csv")











