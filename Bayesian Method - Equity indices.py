#Bayesian Method - measuring conditional probability of increases or decreases in Interest Rates using the interest rate Δ (Delta)

#IMPORTANT: the following code is measuring the probability of an INCREASE in interest rates based on past data, to measure a decrease change code where applicable 
#Otherwise, any classmates who I have reffered here will enjoy this one, all you have to do is paste 'yourdata' into 'yourdata' and run :) If you wanna give me creds that would be cool lol. 

import pandas as pd

def calculate_probability_increase_stock_index(data):
    # YOUR CSV MUST HAVE THESE COLUMNS--> 'Interest Rate' and 'Date'
    data['Stock Index Delta'] = data['Stock Index'].diff()
    data['Stock Index Delta'] = data['Stock Index Delta'].shift(-1)
    
    # intial (prior) probability
    prior_probability_increase = len(data[data['Stock Index Delta'] > 0]) / len(data)
    
    # conditional probability
    probability_increase_given_increase = len(data[(data['Stock Index Delta'] > 0) & (data['Stock Index Delta'].shift(1) > 0)]) / len(data[(data['Stock Index Delta'] > 0)])
    
    # probability of an increase (or decrease)
    probability_increase = (prior_probability_increase * probability_increase_given_increase) / (prior_probability_increase * probability_increase_given_increase + (1 - prior_probability_increase) * (1 - probability_increase_given_increase))
    
    return probability_increase

data = pd.read_csv('yourdata.csv')
probability_increase_stock_index = calculate_probability_increase_stock_index(data)
print('Probability of an increase in stock index:', probability_increase_stock_index)