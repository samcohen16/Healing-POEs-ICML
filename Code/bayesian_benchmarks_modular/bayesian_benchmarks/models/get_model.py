from importlib import import_module
import os

from bayesian_benchmarks.models.non_bayesian_models import non_bayesian_model
from bayesian_benchmarks.models.expert_models   import expert_models 
from bayesian_benchmarks.models.gp_models import gp_models

abs_path = os.path.abspath(__file__)[:-len('/get_model.py')]


def get_regression_model(name):
    print(name)
    
    
    assert name in all_regression_models
    return non_bayesian_model(name, 'regression') or \
               expert_models(name, 'regression') or \
               gp_models(name, 'regression') 
               
        
all_regression_models=[]

all_regression_models.append('expert_100_clustering_minibatching')
all_regression_models.append('expert_100_clustering')
all_regression_models.append('expert_100_random_minibatching')
all_regression_models.append('expert_100_random')
all_regression_models.append('gp')
all_regression_models.append('linear')

all_models = list(set(all_regression_models))