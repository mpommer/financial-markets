# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 21:37:48 2021

@author: marce
"""

import numpy as np
from copy import copy


def LevenberquOptimizer(function, marketPrices, initialGuess, bounds, learningRate=0.1, tol=1e-08, maxIterations = 10000):
    def LevenbergMarquardtStep(function, marketprices, xValue, MSE, learningRateApply):
        gradient = getGradientFiniteDifference(function, xValue)
        # tikhonov regularization
        tikhanovMatrix = gradient.T@gradient + learningRateApply*np.diag(np.diag(gradient.T@gradient))
        
        if np.linalg.matrix_rank(tikhanovMatrix)!= tikhanovMatrix.shape[0]:
            return xValue, MSE
        
        inverse = np.linalg.inv(tikhanovMatrix)
        
        deltaX = inverse@gradient.T@(marketPrice - function(xValue))
        
        xValueCandidate = copy(xValue)
        xValueCandidate[::,1] = xValue[::,1] + deltaX
        functionValueCandidate = function(xValueCandidate)
        newMSE = sum([(x-y)**2 for x,y in zip(functionValueCandidate, marketPrice)])
        
        return xValueCandidate, newMSE
        
    xValue = copy(initialGuess)
    functionValue = function(xValue)
    MSE = sum([(x-y)**2 for x,y in zip(functionValue, marketPrices)])
    
    iteration = 0
    while MSE >tol and iteration <maxIterations:
        learningRateApply = copy(learningRate)
        xValueCandidate, newMSE = LevenbergMarquardtStep(function, marketPrices, xValue, MSE, learningRateApply)
        
        if newMSE < MSE:
            MSE = newMSE
            xValue = xValueCandidate
        else:
            learningRateApply *= 2
            
        if learningRateApply >20:
            return 0
        
        iteration += 1
    
    return xValue, MSE, iteration
            
            

    
def getGradientFiniteDifference(function, volastructure):    
    gradientMatrix = np.zeros((volastructure.shape[0],volastructure.shape[0] ))
    for index, sigma in enumerate(volastructure[::,1]):
        lowerBound = sigma-1e-5
        while lowerBound<0:
            lowerBound /= 10
            
        upperBound = sigma+1e-5
        newvolastructure1 = copy(volastructure)
        newvolastructure1[index][1] = lowerBound
        
        newvolastructure2 = copy(volastructure)
        newvolastructure2[index][1] = upperBound

        gradientMatrix[::,index] = (function(newvolastructure2) - function(newvolastructure1))\
            /(1e-5+sigma-lowerBound)
            
    return gradientMatrix
            
            

def fun(x):    
    price1 = x[0][1]**2 - x[1][1]**2
    price2 = x[1][1]**2
    price3 = x[2][1]**2
    return np.array([price1, price2, price3])


volastructure = np.array([[1.0, 2],
                          [2.0, 2],
                          [3.0, 2]])



grad = getGradientFiniteDifference(fun, volastructure)


marketPrice = np.array([0, 2, 0])

bounds = []
sol, mse, it = LevenberquOptimizer(fun, marketPrice, volastructure, bounds, learningRate=0.1, tol=0.000001)

print('SOl: {} and function valu {}'.format(sol,mse))

print(fun(sol))





