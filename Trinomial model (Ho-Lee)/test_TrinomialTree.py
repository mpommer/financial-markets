# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 10:55:19 2021

@author: marce
"""

import numpy as np
from TrinomialTree import treeConstruction, calculateBondPrice
from tabulate import tabulate


ZeroCurve = np.array([[1., 0.03],
                      [2., 0.04],
                      [3., 0.04],
                      [4., 0.05],
                      [5., 0.06],
                      [6., 0.07]])



tree = treeConstruction(ZeroCurve, lastDate = 5, volatility = 0.001, StepsPerYear=36, a = 0)


paymentTimes = []
cashFlows = []
analyticPrices = []
for index in range(5):
    paymentTimes.append(index+1)
    cashFlows.append(np.array([[index+1, 1.0]]))
    
    analyticPrices.append(np.exp(ZeroCurve[::,1][index]*(-(index+1))))
    
    
ExDates = np.array([[0.0, 0.0]])


numericPrices = []
relativeError = []
for index, cashFlow in enumerate(cashFlows):
    numericPrice = calculateBondPrice(tree, cashFlow, ExDates)
    
    numericPrices.append(numericPrice)
    
    relativeError.append((numericPrice- analyticPrices[index])/analyticPrices[index] * 100)
    

data = np.array([paymentTimes, analyticPrices, numericPrices,relativeError ]).T
colNames = [' Payment Time', ' Ana Price', ' Num Price', ' Rel Error']

print(tabulate(data, headers = colNames))



