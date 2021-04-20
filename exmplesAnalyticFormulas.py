# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:33:41 2021
This class shows how to use the analytic formulas implemented
in "analytic formulas"
@author: Marcel Pommer
"""

from analyticformulas import analyticformulas as af
import numpy as np
np.set_printoptions(precision=4)

nominal = 1000
initialValue = 1.0
strike = 0.9
sigma = 0.7
maturity = 0
discountFactor = 0.95
periodLength = 1.0
analytic = af()
analyticValueCapletBachelier = analytic.bachelierCall(forward = initialValue, optionStrike =strike,
            volatility = sigma, discountFactor = discountFactor, nominal = nominal)

print("Bachelier formula, caplet:             {:.5}".format(analyticValueCapletBachelier));


analyticValueCapletBlackScholes = analytic.blackScholesCall(forward = initialValue, optionStrike =strike,
                            volatility = sigma, discountFactor = discountFactor, nominal = nominal)
		
print("Black Scholes formula, caplet:         {:.5}".format(analyticValueCapletBlackScholes))


analyticValueDigitalCapletBlackScholes = analytic.BlackScholesDigitalCaplet(forward = initialValue, optionStrike =strike,
                            volatility = sigma, discountFactor = discountFactor,nominal = nominal)

print("Black Scholes formula, digital caplet: {:.5}".format(analyticValueDigitalCapletBlackScholes));

#swaption value over three periods
discountFactorSwaption = np.array([0.95, 0.9, 0.85]);

swaptionValue = analytic.BlackScholesSwaption(forward = initialValue, optionStrike =strike,
            volatility = sigma, discountFactor = discountFactorSwaption, optionEnd = 3,nominal = nominal)

print("Black Scholes formula, swaption:       {:.5}".format(swaptionValue));

