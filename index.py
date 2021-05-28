#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 15:42:58 2021

@author: abhijith
"""

from app import app
from app import server
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import dash
import dash_table
import pandas as pd


df = pd.read_csv('classified_metadata.csv')

filename = 'noise_level.csv'
dftoplot = pd.read_csv(filename)
dftoplot.loc[:,'Time'] = pd.to_datetime(dftoplot.loc[:,'Time'])

fig = go.Figure()
fig.add_bar(x=dftoplot['Time'],y=dftoplot['Noise'])
fig.update_layout(xaxis=dict(title='Time'),yaxis=dict(title='Noise (dB)'))

app.layout = html.Div([
     dcc.Graph(figure=fig),
     dash_table.DataTable(id='table',
                          columns=[{"name": i, "id": i} for i in df.columns],
                          data=df.to_dict('records'))
    ])
if __name__ == '__main__':
    app.run_server(debug=True,port=5000,host='0.0.0.0')