#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 20:39:23 2021

@author: abhijith
"""

import dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True