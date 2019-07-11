# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 00:00:14 2019

@author: verlo
"""
PIK='Q_table.p'
with open(PIK, "rb") as f:
    print pickle.load(f)