# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:07:58 2021

@author: marce
"""

import math
import scipy.stats as st


class analyticformulas:
    
    def bachelierCall(self, forward, optionStrike, volatility = 0.05,
                      optionMaturity = 0, periodLength = 1, discountFactor = 1):
        '''
        The function calculates the option (call) price for a normal model 
        (Bachelier).

        Parameters
        ----------
        forward : TYPE double.
            DESCRIPTION. Initial forward value.
        optionStrike : TYPE, double.
            DESCRIPTION. Strike for the option.
        volatility : TYPE, double>0
            DESCRIPTION. The default is 0.05.
        optionMaturity : TYPE, integer.
            DESCRIPTION. The default is 0. Start of the period.
        periodLength : TYPE, int.
            DESCRIPTION. The default is 1.
        discountFactor : Type, double.
            DESCRIPTION. The default is 1 (no discounting).
            If specified defines the disocunt factor.

        Returns
        -------
        TYPE double.
            DESCRIPTION. Disounted option price.

        '''
        if forward == optionStrike :
            return volatility * math.sqrt(optionMaturity / math.pi / 2.0)
        else:    
            dPlus = (forward - optionStrike) / (volatility * math.sqrt(optionMaturity));

            valueAnalytic = periodLength * discountFactor* ((forward - optionStrike) * st.norm.cdf(dPlus, 0.0, 1.0)\
			+ volatility * math.sqrt(optionMaturity) * st.norm.pdf(dPlus, 0.0, 1.0))

            return valueAnalytic
    
    
    def blackScholesCall(self, forward, optionStrike, volatility = 0.05,
                      optionMaturity = 0, periodLength = 1, discountFactor = 1):
        '''
        The function calculates the option (call, Caplet) price for a lognormal model 
        (Black scholes).

        Parameters
        ----------
        forward : TYPE double.
            DESCRIPTION. Initial forward value.
        optionStrike : TYPE, double.
            DESCRIPTION. Strike for the option.
        volatility : TYPE, double>0
            DESCRIPTION. The default is 0.05.
        optionMaturity : TYPE, integer.
            DESCRIPTION. The default is 0. Start of the period.
        periodLength : TYPE, int.
            DESCRIPTION. The default is 1.
        discountFactor : Type, double.
            DESCRIPTION. The default is 1 (no discounting).
            If specified defines the disocunt factor.

        Returns
        -------
        TYPE double.
            DESCRIPTION. Disounted option price.

        '''
        dPlus = (math.log(forward/optionStrike) + 0.5*volatility *volatility *\
                 math.sqrt(optionMaturity))/(volatility * math.sqrt(optionMaturity))
            
        dMinus = dPlus - volatility * optionMaturity
        
        analyticValue = discountFactor * periodLength * (forward * st.norm.cdf(dPlus, 0.0, 1.0) - \
                                optionStrike * st.norm.cdf(dMinus, 0.0, 1.0))
            
        return analyticValue
    
    
    def BlackScholesDigitalCaplet(self, forward, optionStrike, volatility = 0.05,
                      optionMaturity = 0, periodLength = 1, discountFactor = 1):
        
        analyticValue = self.blackScholesCall(forward, -1, volatility,
                            optionMaturity,periodLength,discountFactor)
        
        return analyticValue
    
    def BlackScholesSwaption(self, forward, optionStrike, volatility = 0.05,
            optionMaturity = 0, periodLength = 1, discountFactor = 1, optionEnd = 1):
        # derive swap annuity
        numberOfPeriods = (optionEnd - optionMaturity)/periodLength
        swapAnnuity = 0
        for i in range(numberOfPeriods):
            swapAnnuity += periodLength * discountFactor[i]
        
        analyticValue = self.blackScholesCall(forward, optionStrike, volatility,
                    optionMaturity, discountFactor=swapAnnuity)
        
        return analyticValue
        

