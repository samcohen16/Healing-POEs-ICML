"""
If pytest-xdist is installed, pytest can multiple independent jobs in parallel on a single machine.

Here we use the facility to run experiments.

To run 32 experiments in parallel install xdist (pip install pytest-xdist), then following command can be used

python -m pytest bayesian_benchmarks/scripts/run_all_pytest.py -n 32

"""
import pytest

from bayesian_benchmarks.tasks.regression import run as run_regression


from bayesian_benchmarks.data import regression_datasets
from bayesian_benchmarks.database_utils import Database


all_regression_models = [
    'expert_100_clustering',
    #'gp',
    'linear'
      ]

class ConvertToNamespace(object):
    def __init__(self, adict):
        adict.update({'seed':0,
                      'database_path':''})
        self.__dict__.update(adict)

def check_needs_run(table, d):
    with Database() as db:
        try:
            return (len(db.read(table, ['test_loglik'], d.__dict__)) == 0)
        except:
            return True


@pytest.mark.parametrize('model', all_regression_models)
@pytest.mark.parametrize('dataset', regression_datasets)
@pytest.mark.parametrize('split', range(10))
def test_run_all_regression(model, dataset, split):
    d = ConvertToNamespace({'dataset':dataset,
                            'model' :  model,
                            'split' : split})

    if check_needs_run('regression', d):
        run_regression(d, is_test=False)


