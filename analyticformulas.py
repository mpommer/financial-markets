# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:07:58 2021
Collection of analytic formulas in the context of 
financial markets, for example Black formula and bachelier formula 
for the caplet.
@author: Marcel Pommer
"""

import numpy as np
import math
import scipy.stats as st


class analyticformulas:
    
    def bachelierCall(self, forward, optionStrike, volatility = 0.05,
            optionMaturity = 0, periodLength = 1, discountFactor = 1, nominal = 1):
        '''
        The function calculates the option (call) price for a normal model 
        (Bachelier).

        Parameters
        ----------
        forward : TYPE float.
            DESCRIPTION. Initial forward value.
        optionStrike : TYPE, float.
            DESCRIPTION. Strike for the option.
        volatility : TYPE, float>0
            DESCRIPTION. The default is 0.05.
        optionMaturity : TYPE, integer.
            DESCRIPTION. The default is 0. Start of the period.
        periodLength : TYPE, int.
            DESCRIPTION. The default is 1.
        discountFactor : Type, float.
            DESCRIPTION. The default is 1 (no discounting).
            If specified defines the disocunt factor.
        nominal : Type, float.
            DESCRIPTION. The default is 1.
            If specified defines the nominal.
        Returns
        -------
        TYPE double.
            DESCRIPTION. Disounted option price.

        '''
        if forward == optionStrike :
            return volatility * math.sqrt(optionMaturity / math.pi / 2.0)
        else:
            if optionMaturity > 0:
                dPlus = (forward - optionStrike) / (volatility * math.sqrt(optionMaturity));
            else:
                dPlus = np.Inf
                
            valueAnalytic = periodLength * discountFactor* ((forward - optionStrike) * st.norm.cdf(dPlus, 0.0, 1.0)\
			+ volatility * math.sqrt(optionMaturity) * st.norm.pdf(dPlus, 0.0, 1.0))

            return nominal * valueAnalytic
    
    
    def blackScholesCall(self, forward, optionStrike, volatility = 0.05,
        optionMaturity = 0, periodLength = 1, discountFactor = 1, nominal = 1):
        '''
        The function calculates the option (call, Caplet) price for a lognormal model 
        (Black scholes).

        Parameters
        ----------
        forward : TYPE float.
            DESCRIPTION. Initial forward value.
        optionStrike : TYPE, float.
            DESCRIPTION. Strike for the option.
        volatility : TYPE, float>0
            DESCRIPTION. The default is 0.05.
        optionMaturity : TYPE, integer.
            DESCRIPTION. The default is 0. Start of the period.
        periodLength : TYPE, int.
            DESCRIPTION. The default is 1.
        discountFactor : Type, float.
            DESCRIPTION. The default is 1 (no discounting).
            If specified defines the disocunt factor.
        nominal : Type, float.
            DESCRIPTION. The default is 1.
            If specified defines the nominal.
            
        Returns
        -------
        TYPE double.
            DESCRIPTION. Disounted option price.

        '''
        if optionMaturity > 0:
            dPlus = (math.log(forward/optionStrike) + 0.5*volatility *volatility *\
                 math.sqrt(optionMaturity))/(volatility * math.sqrt(optionMaturity))
        else:
            dPlus = np.Inf
            
        dMinus = dPlus - volatility * optionMaturity
        
        analyticValue = discountFactor * periodLength * (forward * st.norm.cdf(dPlus, 0.0, 1.0) - \
                                optionStrike * st.norm.cdf(dMinus, 0.0, 1.0))
            
        return nominal * analyticValue
    
    
    def BlackScholesDigitalCaplet(self, forward, optionStrike, volatility = 0.05,
        optionMaturity = 0, periodLength = 1, discountFactor = 1, nominal = 1):
        '''
        The function calculates the option (digital Caplet) price for a lognormal model 
        (Black scholes). A digital caplet is a derivate which returns 1 if the option price 
        is above the strike a option end T and returns 0 otherwise. In order to
        increase/decrease the payoff the user can multiply the price with a 
        nominal.

        Parameters
        ----------
        forward : TYPE float.
            DESCRIPTION. Initial forward value.
        optionStrike : TYPE, float.
            DESCRIPTION. Strike for the option.
        volatility : TYPE, float>0
            DESCRIPTION. The default is 0.05.
        optionMaturity : TYPE, integer.
            DESCRIPTION. The default is 0. Start of the period.
        periodLength : TYPE, int.
            DESCRIPTION. The default is 1.
        discountFactor : Type, float.
            DESCRIPTION. The default is 1 (no discounting).
            If specified defines the disocunt factor.
        nominal : Type, float.
            DESCRIPTION. The default is 1.
            If specified defines the nominal.
            
        Returns
        -------
        TYPE double.
            DESCRIPTION. Disounted option price.

        '''
        analyticValue = self.blackScholesCall(0, -1, volatility,
                            optionMaturity,periodLength,discountFactor, nominal=nominal)
        
        return analyticValue
    
    
    def BlackScholesSwaption(self, forward, optionStrike, volatility = 0.05,
    optionMaturity = 0, periodLength = 1, discountFactor = 1, optionEnd = 1, nominal = 1):
        '''
        The function calculates the swaption price for a lognormal model 
        (Black scholes). A value of a swaption is defined as 
        $v_{swaption} = max(value_{swap}(T_1), 0)$.
        
        Parameters
        ----------
        forward : TYPE float.
            DESCRIPTION. Inital value of the swaption.
        optionStrike : TYPE float.
            DESCRIPTION. Strike of the swaption.
        volatility : TYPE, float>0.
            DESCRIPTION. Volatility of the model. The default is 0.05.
        optionMaturity : TYPE, optional (float/int).
            DESCRIPTION. Start of the Swpation. The default is 0.
        periodLength : TYPE, optional (float/int).
            DESCRIPTION. The default is 1.
        discountFactor : TYPE, float.
            DESCRIPTION. The default is 1 (no discounting).
        optionEnd : TYPE, optional (float/int).
            DESCRIPTION. The default is 1. End of the swaption.
        nominal : Type, float.
            DESCRIPTION. The default is 1.
            If specified defines the nominal.
            
        Returns
        -------
        analyticValue : TYPE float.
            DESCRIPTION. Analytic value for a swaption under black scholes model 
            (lognormal).

        '''
        # check if discountFactor is iterable
        numberOfPeriods = (optionEnd - optionMaturity)/periodLength
        try:
            x = numberOfPeriods.is_integer()
                        
        except ValueError:
            print("Number of Periods is no integer")
            return np.inf
            
        discountArray = np.array(discountFactor)
        if discountArray.size != numberOfPeriods:
            x = np.arange(numberOfPeriods, dtype=int)
            x = np.full_like(x, discountFactor, dtype=np.double)
            discountFactor = x
            
       
        # derive swap annuity
        swapAnnuity = 0
        for i in range(int(numberOfPeriods)):
            swapAnnuity += periodLength * discountFactor[i]
        
        analyticValue = self.blackScholesCall(forward, optionStrike, volatility,
                    optionMaturity, discountFactor=swapAnnuity, nominal = nominal)
        
        return analyticValue
        

