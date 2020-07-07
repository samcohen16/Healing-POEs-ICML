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


def expert_predictions(X_test, model, gather, mu_s = None, var_s = None, minibatching=False, batch_size = 10000):
    """Predicting aggregated mean and variance for all test points
            
    Inputs : 
            -- X_test, dimension: n_test_points x dim_x : Test points 
            -- model : Model object to predict with
            -- gather : True/False : Set to true if we want to gather predictions at X_test, and to False if we want to aggregate predictions at X_test
            -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test points X_test
            -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test points X_test
            -- minibatching, True/False : Set to True if minibatching is desired
            -- batch_size, int : size of minibatches, default to 10000
            
    Output : 
        if gather = True:
            -- m, dimension: n_experts x n_test_points : predictive mean of each expert at each test points X_test
            -- v, dimension: n_experts x n_test_points : predictive variance of each expert at each test points X_test
        if gather = False:
            -- m, dimension: n_test_points x 1 : aggregated predictive mean
            -- v, dimension: n_test_points x 1 : aggregated predictive variance
            
    """
    if minibatching:

                    # Set batch size and compute number of batches         
                    num_batches = math.ceil(X_test.shape[0]/batch_size)

                    for i in tqdm(range(num_batches)):

                        
                        start_i, end_i = i*batch_size, (i+1)*batch_size
                        
                        # Gather the predictions of all experts at a specific minibatch  - m_i, v_i are n_experts x n_test_points_in_batch
                        if gather:
                            m_i, v_i = model.gather_predictions(X_test[start_i : end_i])
                        # Aggregate the predictions of all experts at a specific minibatch  - m_i, v_i are n_test_points_in_batch x 1 and
                        # mu_s, var_s are n_experts x n_test_points
                        else: 
                            m_i, v_i = model.predict(X_test[start_i : end_i], mu_s[:,start_i : end_i], var_s[:,start_i : end_i])
                            
                        if i==0:
                            m = m_i
                            v = v_i
                        else:
                            if gather:
                                #Append gathered predictions of all experts at minibatch i. m_i is n_experts x n_test_points_in_batch_i
                                m = tf.concat((m,m_i),1)
                                v = tf.concat((v,v_i),1)
                            else:
                                #Append aggregated predictions of all experts at minibatch i. m_i is n_test_points_in_batch_i x 1
                                m = tf.concat((m,m_i),0)
                                v = tf.concat((v,v_i),0)

    else:
                    # Gather the predictions of all experts at all test points without minibatching  
                    if gather:
                        m, v = model.gather_predictions(X_test)
                    # Aggregate the predictions of all experts at all test points without minibatching  

                    else:
                        m, v = model.predict(X_test, mu_s, var_s)
    
    return(m,v)


def update_score_database(m, v, data, ARGS, is_test, power = None, weighting = None, model_name = None):
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
    
    if 'expert' in ARGS.model:
        res['model']=model_name+'_'+str(power)+'_'+ARGS.model.split('_')[1]+'_'+ARGS.model.split('_')[2]+'_'+weighting

    if not is_test:  # pragma: no cover
        with Database(ARGS.database_path) as db:
            db.write('regression', res)
    print('end', res)
    
    return(res)
    
    
def run(ARGS, data=None, model=None, is_test=False):
    
    # Set list of softmax scaling we want to train experts with
    powers = [100]
    
    # Set list of models (and their weighting methods) to be trained
    dict_models={'bar':['variance'], 'gPoE':['uniform', 'variance'], 'rBCM':['diff_entr', 'variance'], 'BCM':['no_weights'], 'PoE':['no_weights']}
    
    # Gather the data
    data = data or get_regression_data(ARGS.dataset, split=ARGS.split)
    print(data.X_train.shape)
    

    # Initialize the model
    model = model or get_regression_model(ARGS.model)(is_test=is_test, seed=ARGS.seed)
    if (ARGS.model == 'gp' and data.N>6000):
        return('too large data for full gp')
    
    # Optimize the model by maximizing sum of log-marginal likelihoods
    print('model fitting')
    model.fit(data.X_train, data.Y_train)
    
    
        
    
    if 'expert' in ARGS.model:
        
        if 'minibatching' in ARGS.model:
            minibatching = True
        else:
            minibatching = False
        
        # Gather the predictions of all experts at all test inputs with an option to minibatch. mu_s, var_s are n_expert x n_test
        print('gathering predictions')
        mu_s, var_s = expert_predictions(data.X_test, model, minibatching = minibatching, gather = True)
                    
      
        
        # Loop over models (Poe,...), weighting methods (Wass,variance,...) and powers  (softmax scaling)
        for model_name in dict_models.keys():
            for weighting in dict_models[model_name]:
                for power in powers:
                    model.power = power
                    model.model = model_name
                    model.weighting = weighting
                    
                    #Aggregate predictions for a single model (using a specific weighting scheme, e.g gPoE_var with T=100). m,v are n_test x 1
                    print('prediction aggregation')
                    m, v = expert_predictions(data.X_test, model, mu_s = mu_s, var_s = var_s, minibatching = minibatching, gather = False)
                    
                    #Add scores (RMSE/NLPD) of the single model to the database
                    res = update_score_database(m, v, data, ARGS, is_test, power = power, weighting = weighting, model_name = model_name)
                    
                    if weighting in [ 'no_weights','uniform','diff_entr'] :
                        break
    else:
                   
                m, v = model.predict(data.X_test)
                
                res = update_score_database(m, v, data, ARGS, is_test)

                

    return res


if __name__ == '__main__':
    run(parse_args())
    
   



