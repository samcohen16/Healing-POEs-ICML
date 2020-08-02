import gpflow
import tensorflow as tf
import numpy as np
from gpflow.utilities import print_summary
import multiprocessing as mp
from tqdm import tqdm
from sklearn.cluster import KMeans


def compute_weights(mu_s, var_s, power, weighting, prior_var=None, softmax=False):
    
    """ Compute unnormalized weight matrix
    Inputs : 
            -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
            -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
            -- power, dimension : 1x1 : Softmax scaling
            -- weighting, str : weighting method (variance/wass/uniform/diff_entr/no_weights)
            -- prior_var, dimension: 1x1 : shared prior variance of expert GPs
            -- soft_max_wass : logical : whether to use softmax scaling or fraction scaling
            
    Output : 
            -- weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of ith expert at jth test point
    """
    
    if weighting == 'variance':
        weight_matrix = tf.math.exp(-power * var_s)

    if weighting == 'wass':
        wass = mu_s**2 + (var_s - prior_var)**2
        
        if softmax == True:
            weight_matrix = tf.math.exp(power * wass)
        else:
            weight_matrix = wass**power

    if weighting == 'uniform':
        weight_matrix = tf.ones(mu_s.shape, dtype = tf.float64) / mu_s.shape[0]

    if weighting == 'diff_entr':
        weight_matrix = 0.5 * (tf.math.log(prior_var) - tf.math.log(var_s))
        
    if weighting == 'no_weights':
        weight_matrix = 1
    
    weight_matrix = tf.cast(weight_matrix, tf.float64)
    

    return weight_matrix



def normalize_weights(weight_matrix):
    """ Compute unnormalized weight matrix
    Inputs : 
            -- weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of ith expert at jth test point
            
            
    Output : 
            -- weight_matrix, dimension: n_expert x n_test_points : normalized weight of ith expert at jth test point
    """
    
    sum_weights = tf.reduce_sum(weight_matrix, axis=0)
    weight_matrix = weight_matrix / sum_weights
    
    return weight_matrix

    
def regression_model(points_per_experts, partition_type):
    class RegressionModel:

        def __init__(self, is_test=False, seed=0):


            self.partition_type = partition_type
            self.points_per_experts = points_per_experts

            self.opt = gpflow.optimizers.Scipy()

        def fit(self, X: np.ndarray, Y: np.ndarray):
            
            """ Initiate the individual experts and fit their shared hyperparameters by 
                                minimizing the sum of negative log marginal likelihoods
            
    Inputs : 
            -- X, dimension: n_train_points x dim_x : Training inputs
            -- Y, dimension: n_train_points x 1 : Training Labels
            
    """
            # Compute number of experts 
            self.M = int(np.max([int(X.shape[0]) / self.points_per_experts, 1]))
            
            # Compute number of points experts 
            self.N = int(X.shape[0] / self.M)
            
            # If random partition, assign random subsets of data to each expert
            if self.partition_type == 'random':
                self.partition = np.random.choice(X.shape[0], size=(self.M, self.N), replace=False)
            
            # If clustering partition, assign fit a K_means to the train data and assign a cluster to each expert
            if self.partition_type == 'clustering':
                kmeans = KMeans(n_clusters=self.M).fit(X[:10000])
                labels = kmeans.predict(X)

                self.partition = []

                for m in range(self.M):
                    self.partition.append(np.where(labels == m)[0])

            self.kern = gpflow.kernels.RBF()
            
            # Initialize each expert along with their individual subset of data and a shared kernel
            self.experts = [gpflow.models.GPR(data=(X[self.partition[i]], Y[self.partition[i]]),
                                              kernel=self.kern, mean_function=None) for i in range(self.M)]

            self.likelihood = gpflow.likelihoods.Gaussian()

            for expert in self.experts:
                expert.likelihood = self.likelihood
            
            # Jointly optimize the expert's set of shared hyperparameters
            self.optimize()

            
        def tot_negloglike(self):
            """Computing the sum of the negative log marginal likelihoods of all experts
            
    Outputs : 
            -- negloglik, dimension: 1 x 1 : sum of negative log marginal likelihood
            
            
    """
            neg_loglik = tf.reduce_sum([-expert.log_marginal_likelihood() for expert in self.experts])
            print(neg_loglik)
            return neg_loglik

        def optimize(self, max_iter=200):
            """Joint optimization of expert model's shared hyperparameters. 
            
    Inputs : 
            -- max_iter, dimension: 1 x 1 : number of opt iterations
            
            
    """
            
            opt_logs = self.opt.minimize(self.tot_negloglike,
                                         (*self.kern.trainable_variables,   *self.likelihood.trainable_variables),
                                         options=dict(maxiter=max_iter), compile=False)
        
        def predict(self, xt_s, mu_s, var_s):
            """Predicting aggregated mean and variance for all test points
            
    Inputs : 
            -- xt_s, dimension: n_test_points x dim_x : Test points 
            -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
            -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
            
    Output : 
            -- mu, dimension: n_test_points x 1 : aggregated predictive mean
            -- var, dimension: n_test_points x 1 : aggregated predictive variance
            
    """
            return self.prediction_aggregation(xt_s, mu_s, var_s, self.model, power=self.power,
                                               weighting=self.weighting)

        def gather_predictions(self, xt_s):
            """Gathering the predictive means and variances of all local experts at all test points
            
    Inputs : 
            -- xt_s, dimension: n_test_points x dim_x : Test points 
            
    Output : 
            -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
            -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
    """
            # Gather the predictive means and variances of each experts 
            #                  (a list with the means and variances of each expert - len(list)=num_experts )
            predictive = [expert.predict_f(xt_s) for expert in self.experts]
            
            #Creating a list of means and a list of variances
            mu_s, var_s = zip(*predictive)
            
            #Stacking so that mu_s and var_s are tf tensors of dim n_expert x n_test_points 
            mu_s = tf.stack(mu_s)[:, :, 0]
            var_s = tf.stack(var_s)[:, :, 0]
            return mu_s, var_s

        def prediction_aggregation(self, xt_s,mu_s,var_s, method='PoE', weighting='wass', power=8):

            """ Aggregation of predictive means and variances of local experts
            
    Inputs : 
            -- xt_s, dimension: n_test_points x dim_x : Test points 
            -- mu_s, dimension: n_expert x n_test_points : predictive mean of each expert at each test point
            -- var_s, dimension: n_expert x n_test_points : predictive variance of each expert at each test point
            -- method, str : aggregation method (PoE/gPoE/BCM/rBCM/bar)
            -- weighting, str : weighting method (variance/wass/uniform/diff_entr/no_weights)
            -- power, dimension : 1x1 : Softmax scaling
            
    Output : 
            -- mu, dimension: n_test_points x 1 : aggregated predictive mean
            -- var, dimension: n_test_points x 1 : aggregated predictive variance
    """
            # Compute prior variance (shared between all experts)
            prior_var = self.kern(xt_s[0], xt_s[0])
            
            # Compute individual precisions - dim: n_experts x n_test_points
            prec_s = 1/var_s
            
            # Compute weight matrix - dim: n_experts x n_test_points
            weight_matrix = compute_weights(mu_s, var_s, power, weighting, prior_var)

            
            # For all DgPs, normalized weights of experts requiring normalized weights and compute the aggegated local precisions
            if method == 'PoE':
                            
                prec = tf.reduce_sum(prec_s, axis=0)                

            if method == 'gPoE':
                
                weight_matrix = normalize_weights(weight_matrix)

                prec = tf.reduce_sum(weight_matrix * prec_s, axis=0)
                

            if method == 'BCM':
                
                prec = tf.reduce_sum(prec_s, axis=0) + (1 - self.M) / prior_var 

            if method == 'rBCM':
                
                prec = tf.reduce_sum(weight_matrix * prec_s, axis=0) \
                       + (1 - tf.reduce_sum(weight_matrix, axis=0)) / prior_var
                
                

            #Compute the aggregated predictive means and variance of the barycenter    
            if method == 'bar':
                
                weight_matrix = normalize_weights(weight_matrix)
                
                mu = tf.reduce_sum(weight_matrix * mu_s, axis=0)
                var = tf.reduce_sum(weight_matrix * var_s, axis=0)
            
            #For all DgPs compute the aggregated predictive means and variance    
            else:
                
                var = 1 / prec
                                    
                mu = var * tf.reduce_sum(weight_matrix * prec_s * mu_s, axis=0)
            
            mu = tf.reshape(mu, (-1, 1))
            var = tf.reshape(var, (-1, 1))
            
            return self.lik_aggregation(mu, var)

        def lik_aggregation(self, mu, var):

            return (mu, var + self.likelihood.variance)

    return RegressionModel


def expert_models(name, task):
    if 'expert_' in name:
        points_per_experts = int(name.split('_')[1])

        partitioning = name.split('_')[2]

        return regression_model(points_per_experts, partitioning)