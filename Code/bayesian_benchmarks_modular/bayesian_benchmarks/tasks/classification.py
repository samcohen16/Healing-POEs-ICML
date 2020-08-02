"""
A classification task, which can be either binary or multiclass.

Metrics reported are test loglikelihood, classification accuracy. Also the predictions are stored for 
analysis of calibration etc. 

"""

import sys
sys.path.append('../')

import argparse
import numpy as np

from scipy.stats import multinomial

from bayesian_benchmarks.data import get_classification_data
from bayesian_benchmarks.models.get_model import get_classification_model
from bayesian_benchmarks.database_utils import Database
import tensorflow as tf
import math
from tqdm import tqdm

def parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='variationally_sparse_gp', nargs='?', type=str)
    parser.add_argument("--dataset", default='statlog-german-credit', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--seed", default=0, nargs='?', type=int)
    parser.add_argument("--database_path", default='', nargs='?', type=str)
    return parser.parse_args()


def top_n_accuracy(preds, truths, n):
        best_n = np.argsort(preds, axis=1)[:,-n:]
       
        ts = np.argmax(truths, axis=1)

        successes = 0
        for i in range(ts.shape[0]):
            if ts[i] in best_n[i,:]:
                successes += 1
        return float(successes)/ts.shape[0]
    
def onehot(Y, K):
        return np.eye(K)[Y.flatten().astype(int)].reshape(Y.shape[:-1]+(K,))
    
    
def run(ARGS, data=None, model=None, is_test=False):

    powers = [10]
    dict_models={'bar':['variance'],'gPoE':['uniform','variance'],'rBCM':['diff_entr','variance'],'BCM':['no_weights'],'PoE':['no_weights']}
    data = data or get_classification_data(ARGS.dataset, split=ARGS.split)
    model = model or get_classification_model(ARGS.model)(data.K, is_test,seed=ARGS.seed)

    
    Y_oh = onehot(data.Y_test, data.K)[None, :, :]  # 1, N_test, K
    

    

    print('model fitting')
    model.fit(data.X_train, data.Y_train)

    if 'expert' in ARGS.model:
        
        print('gathering predictions')
        
        
        mu_s, var_s = model.gather_predictions(data.X_test)
                
                
        print('prediction aggregation')
        
        for model_name in dict_models.keys():
            for weighting in dict_models[model_name]:
                for power in powers:
                        model.power = power
                        model.model = model_name
                        model.weighting = weighting
                        p = model.predict(data.X_test, mu_s, var_s)

                        res = {}

                        # clip very large and small probs
                        eps = 1e-12
                        p = np.clip(p, eps, 1 - eps)
                        p = p / np.expand_dims(np.sum(p, -1), -1)
                        
                        logp = multinomial.logpmf(Y_oh, n=1, p=p)

                        res['test_loglik'] = np.average(logp)

                        res['top_1_acc'] = top_n_accuracy(p,np.reshape(Y_oh, (-1,data.K)), 1)
                        res['top_2_acc'] = top_n_accuracy(p,np.reshape(Y_oh, (-1,data.K)), 2)
                        res['top_3_acc'] = top_n_accuracy(p,np.reshape(Y_oh, (-1,data.K)), 3)

                        pred = np.argmax(p, axis=-1)
    
                      
                        res.update(ARGS.__dict__)

                        res['model']=model_name+'_'+str(power)+'_'+ARGS.model.split('_')[1]+'_'+ARGS.model.split('_')[2]+'_'+weighting

                        print('end', res)
                        if not is_test:  # pragma: no cover
                            with Database(ARGS.database_path) as db:
                                db.write('classification', res)
                        
                    
                        if weighting in [ 'no_weights','uniform','diff_entr']:
                            break
    else:

                p = model.predict(data.X_test) # N_test, 

               # clip very large and small probs
                eps = 1e-12
                p = np.clip(p, eps, 1 - eps)
                p = p / np.expand_dims(np.sum(p, -1), -1)

                # evaluation metrics
                res = {}

                logp = multinomial.logpmf(Y_oh, n=1, p=p)
                res['test_loglik'] = np.average(logp)
                res['top_1_acc'] = top_n_accuracy(p,np.reshape(Y_oh,(-1,data.K)),1)
                res['top_2_acc'] = top_n_accuracy(p,np.reshape(Y_oh,(-1,data.K)),2)
                res['top_3_acc'] = top_n_accuracy(p,np.reshape(Y_oh,(-1,data.K)),3)
                pred = np.argmax(p, axis=-1)


                res.update(ARGS.__dict__)

                if not is_test:  # pragma: no cover
                    with Database(ARGS.database_path) as db:
                        db.write('classification', res)

    return res 
                
   


if __name__ == '__main__':
    run(parse_args())