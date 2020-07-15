from importlib import import_module
import os

from bayesian_benchmarks.models.non_bayesian_models import non_bayesian_model
from bayesian_benchmarks.models.expert_models   import expert_models 
from bayesian_benchmarks.models.gp_models import gp_models
from bayesian_benchmarks.models.expert_models_var_class import expert_models as expert_models_class 

abs_path = os.path.abspath(__file__)[:-len('/get_model.py')]


def get_regression_model(name):
    print(name)
    
    
    assert name in all_regression_models
    return non_bayesian_model(name, 'regression') or \
               expert_models(name, 'regression') or \
               gp_models(name, 'regression')

def get_classification_model(name):
    assert name in all_classification_models
    return expert_models_class(name, 'classification')
          

all_regression_models=[]                
               
        
all_regression_models=[]

all_regression_models.append('expert_100_clustering')
all_regression_models.append('expert_100_random')
all_regression_models.append('expert_200_clustering')
all_regression_models.append('expert_200_random')
all_regression_models.append('expert_500_clustering')
all_regression_models.append('expert_500_random')
all_regression_models.append('gp')
all_regression_models.append('linear')

all_classification_models = [
    'expert_500_random',
    #'expert_800_clustering'
]

all_models = list(set(all_regression_models).union(set(all_classification_models)))

