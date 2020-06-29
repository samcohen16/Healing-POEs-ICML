"""
A conditional Gaussian estimation task: model p(y_n|x_n) = N(a(x_n), b(x_n))
Metrics reported are test log likelihood, mean squared error, and absolute error, all for normalized and unnormalized y.
"""

import argparse
import numpy as np
from scipy.stats import norm
import tensorflow as tf
from bayesian_benchmarks.data import get_regression_data
from bayesian_benchmarks.database_utils import Database
from bayesian_benchmarks.models.get_model import get_regression_model
import math
from tqdm import tqdm
def parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='linear', nargs='?', type=str)
    parser.add_argument("--dataset", default='energy', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--seed", default=0, nargs='?', type=int)
    parser.add_argument("--database_path", default='', nargs='?', type=str)
    return parser.parse_args()

def run(ARGS, data=None, model=None, is_test=False):
    
    # Set list of softmax scaling we want to train experts with
    powers = [1, 10, 15,25,35,50,75, 100,150]
    
    # Set list of models (and their weighting methods) to be trained
    dict_models={'bar':['variance'],'gPoE':['uniform','variance'],'rBCM':['diff_entr','variance'],'BCM':['no_weights'],'PoE':['no_weights']}
    
    # Gather the data
    data = data or get_regression_data(ARGS.dataset, split=ARGS.split)
    print(data.X_train.shape)
    

    # Initialize the model
    model = model or get_regression_model(ARGS.model)(is_test=is_test, seed=ARGS.seed)
    if (ARGS.model == 'gp' and data.N>10000):
        return('too large data for full gp')
    
    # Optimize the model by maximizing sum of log-marginal likelihoods
    print('model fitting')
    model.fit(data.X_train, data.Y_train)
    
    
        
    
    if 'expert' in ARGS.model:
        
        # Gather the predictions of all experts at all test inputs with an option to minibatch
        print('gathering predictions')
        
        
        # Gather the predictions of all experts at all test points without minibatching  
        mu_s, var_s = model.gather_predictions(data.X_test)
                
        # Aggregate predictions of all experts       
        print('prediction aggregation')
        
        # Loop over models (Poe,...), weighting methods (Wass,variance,...) and powers  (softmax scaling)
        for model_name in dict_models.keys():
            for weighting in dict_models[model_name]:
                for power in powers:
                    model.power = power
                    model.model = model_name
                    model.weighting = weighting
                    
                    #Aggregate predictions for a single mode
                    m,v = model.predict(data.X_test,mu_s,var_s)
                    
                    
                    res = {}

                    l = norm.logpdf(data.Y_test, loc=m, scale=v**0.5)
                    res['test_loglik'] = np.average(l)

                    lu = norm.logpdf(data.Y_test * data.Y_std, loc=m * data.Y_std, scale=(v**0.5) * data.Y_std)
                    res['test_loglik_unnormalized'] = np.average(lu)

                    d = data.Y_test - m
                    du = d * data.Y_std

                    res['test_mae'] = np.average(np.abs(d))
                    res['test_mae_unnormalized'] = np.average(np.abs(du))

                    res['test_rmse'] = np.average(d**2)**0.5
                    res['test_rmse_unnormalized'] = np.average(du**2)**0.5

                    res.update(ARGS.__dict__)

                    
                    res['model']=model_name+'_'+str(power)+'_'+ARGS.model.split('_')[1]+'_'+ARGS.model.split('_')[2]+'_'+weighting

                    print('end', res)
                    if not is_test:  # pragma: no cover
                        with Database(ARGS.database_path) as db:
                            db.write('regression', res)
                    if weighting in [ 'no_weights','uniform','diff_entr'] :
                        break
    else:
                   
                m, v = model.predict(data.X_test)

                res = {}

                l = norm.logpdf(data.Y_test, loc=m, scale=v**0.5)
                res['test_loglik'] = np.average(l)

                lu = norm.logpdf(data.Y_test * data.Y_std, loc=m * data.Y_std, scale=(v**0.5) * data.Y_std)
                res['test_loglik_unnormalized'] = np.average(lu)

                d = data.Y_test - m
                du = d * data.Y_std

                res['test_mae'] = np.average(np.abs(d))
                res['test_mae_unnormalized'] = np.average(np.abs(du))

                res['test_rmse'] = np.average(d**2)**0.5
                res['test_rmse_unnormalized'] = np.average(du**2)**0.5

                res.update(ARGS.__dict__)

                if not is_test:  # pragma: no cover
                    with Database(ARGS.database_path) as db:
                        db.write('regression', res)
                print('end', res)

    return res


if __name__ == '__main__':
    run(parse_args())