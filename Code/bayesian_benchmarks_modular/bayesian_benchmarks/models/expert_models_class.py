import gpflow
import tensorflow as tf
import numpy as np
from gpflow.utilities import print_summary
import multiprocessing as mp
from tqdm import tqdm
from sklearn.cluster import KMeans


def compute_weights(mu_s, var_s, power, weighting, prior_var=None, softmax_wass=False):


    """ Compute unnormalized weight matrix
    Inputs : 
            -- mu_s, dimension: n_expert x n_test_points : predictive latent gp mean of each expert at each test point
            -- var_s, dimension: n_expert x n_test_points : predictive latent gp variance of each expert at each test point
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
        wass = mu_s ** 2 + (var_s - prior_var)**2     
        
        if softmax == True:
            weight_matrix = tf.math.exp(power * wass)
        else:
            weight_matrix = wass**power

    if weighting == 'uniform':
        weight_matrix = tf.ones(mu_s.shape, dtype = tf.float64) / mu_s.shape[0]

    if weighting == 'diff_entr':
        weight_matrix = 0.5 * (tf.math.log(prior_var) - tf.math.log(var_s))
        
    if weighting == 'softmax_diff_entr':
        diff_entr = 0.5 * (tf.math.log(prior_var) - tf.math.log(var_s))
        
        if softmax == True:
            weight_matrix = tf.math.exp(power * diff_entr)
        else:
            weight_matrix = diff_entr**power
        
    if weighting == 'no_weights':
        weight_matrix = 1
    
    weight_matrix = tf.cast(weight_matrix, tf.float64)
    

    return weight_matrix



def normalize_weights(weight_matrix):
    """ Compute unnormalized weight matrix
    Inputs : 
            -- weight_matrix, dimension: n_expert x n_test_points : unnormalized weight of ith expert at jth test points
    Output : 
            -- weight_matrix, dimension: n_expert x n_test_points : normalized weight of ith expert at jth test point
    """
    
    sum_weights = tf.reduce_sum(weight_matrix, axis=0)
    weight_matrix = weight_matrix / sum_weights
    
    return (weight_matrix)


def classification_model(points_per_experts, partition_type):
    class ClassificationModel:

        def __init__(self,C, is_test=False, seed=0):
            """
            Initialising a classification expert model object
            Inputs:  C , dimension: 1 x 1 int for number of classes C
                        is_test , Boolean test run flag is_test
                        seed , dimension: 1 X 1 int , current randomisation seed 
                        """

            self.C = C

            N = None
            kern = None

            self.partition_type = partition_type
            self.kern = kern
            self.N = N
            self.points_per_experts = points_per_experts

           

        def fit(self, X: np.ndarray, Y: np.ndarray):

            """ Initiate the individual experts and fit their shared hyperparameters by 
                                minimizing the sum of negative ELBOs 
            
            Inputs : 
            -- X, dimension: n_train_points x dim_x : Training inputs
            -- Y, dimension: n_train_points x 1 : Training Labels
            
            """

            self.ind = 100
            self.X = X
            self.Y = Y

           

            self.M = int(np.max([int(X.shape[0]) / self.points_per_experts, 1]))
            self.partition_type = partition_type
            self.centers =None
            

            
            if self.N is None:
                self.N = int(X.shape[0] / self.M)

           
            self.partition  = np.random.choice(X.shape[0],size=(self.M, self.N),replace=False)
            
            if self.partition_type == 'clustering':
                
                kmeans = KMeans(n_clusters=self.M).fit(self.X[:10000])
                labels = kmeans.predict(self.X)

                self.partition = []

                for m in range(self.M):
                    self.partition.append(np.where(labels == m)[0])
            
            
            if self.kern is None:
                self.kern = gpflow.kernels.RBF()
          

            self.invlink = gpflow.likelihoods.RobustMax(self.C)  
            self.likelihood = gpflow.likelihoods.MultiClass(self.C,invlink=self.invlink)



            self.experts = [gpflow.models.SVGP(kernel =self.kern,likelihood=self.likelihood,num_latent_gps=self.C,inducing_variable=X[self.partition[i][0:self.ind]].copy()) for i in range(self.M)]

            self.opt = gpflow.optimizers.Scipy()

            

            self.optimize()

        def predict(self, Xs,mu_s,var_s):

            """Inputs : 
            -- xt_s, dimension: n_test_points x dim_x : Test points 
            --mu_s, dimension: n_expert x n_test_points x  n_classes : predictive latent gp means of each expert at each test point
            -- var_s, dimension: n_expert x n_test_points x  n_classes : predictive latent gp variances of each expert at each test point
            
            Output : 
            -- mu, dimension: n_test_points x n_classes : aggregated predictive mean
            """

            return self.prediction_aggregation(Xs, mu_s, var_s, self.model, power=self.power, weighting=self.weighting)

        def tot_neg_elbo(self):

            """Computing the sum of negative evidence lower bounds (ELBOs) of all experts
            
            Outputs : 
            -- negloglik, dimension: 1 x 1 : sum of negative ELBOs
             """
            neg_elbo = tf.reduce_sum([-self.experts[i].elbo((self.X[self.partition[i]],self.Y[self.partition[i]])) for i in range(self.M)])
            print(neg_elbo)
            return neg_elbo

        def optimize(self,max_iter=150):

            """Joint optimization of expert model's shared hyperparameters. 
            
            Inputs : 
                    -- max_iter, dimension: 1 x 1 : number of opt iterations
            
            """

            opt_logs = self.opt.minimize(self.tot_neg_elbo,sum([expert.trainable_variables for expert in self.experts],()),options=dict(maxiter=max_iter))

        def gather_predictions(self, xt_s):

            """Gathering the predictive latent gp means and variances of all local experts at all test points
            
            Inputs : 
                    -- xt_s, dimension: n_test_points x dim_x : Test points 
            
            Output : 
                    -- mu_s, dimension: n_expert x n_test_points x  n_classes : predictive latent gp means of each expert at each test point
                    -- var_s, dimension: n_expert x n_test_points x  n_classes : predictive latent gp variances of each expert at each test point"""

            predictive = [expert.predict_f(xt_s) for expert in self.experts]
            mu_s,var_s = zip(*predictive)
            mu_s = tf.stack(mu_s)
            var_s = tf.stack(var_s)
            return mu_s,var_s

        def prediction_aggregation(self, xt_s,mu_s,var_s, method='PoE', weighting='wass', power=8):
                """ Aggregation of predictive means and variances of local experts
            
                 Inputs : 
                    -- xt_s, dimension: n_test_points x dim_x x  n_classes: Test points 
                    -- mu_s, dimension: n_expert x n_test_points x  n_classes : predictive latent gp means of each expert at each test point
                    -- var_s, dimension: n_expert x n_test_points x  n_classes : predictive latent gp variances of each expert at each test point
                    -- method, str : aggregation method (PoE/gPoE/BCM/rBCM/bar)
                    --- weighting, str : weighting method (variance/wass/uniform/diff_entr/no_weights)
                    -- power, dimension : 1x1 : Softmax scaling
            
                Output : 
                    -- mu, dimension: n_test_points x  n_classes : aggregated predictive mean
                """
                
                nt = xt_s.shape[0]
                mu = np.zeros([nt, self.C],dtype='float64')
                var = np.zeros([nt, self.C],dtype='float64')

                prior_var = self.experts[0].kernel(xt_s[0], xt_s[0])

                
                #Process each latent gp individually 
                for j in range(self.C):
                    
                    mu_s_c = mu_s[:, :, j]
                    var_s_c = var_s[:, :, j]
                    
                    weight_matrix = compute_weights(mu_s_c, var_s_c, power, weighting, prior_var)

                    prec_s= 1/var_s_c

                    if method == 'PoE':
                                    
                        prec = tf.reduce_sum(prec_s, axis=0)
                        #var[:,j] = 1 / prec
                        

                    if method == 'gPoE':
                        
                        weight_matrix = normalize_weights(weight_matrix)

                        prec = tf.reduce_sum(weight_matrix * prec_s , axis=0)
                        

                    if method == 'BCM':
                        
                        prec = tf.reduce_sum(prec_s, axis=0) + (1 - self.M) / prior_var 

                    if method == 'rBCM':
                        
                        
                        prec = tf.reduce_sum(weight_matrix * prec_s, axis=0) \
                            + (1 - tf.reduce_sum(weight_matrix, axis=0)) / prior_var
                            
                           
                        
                    if method != 'bar':
                        
                        var[:, j] = 1 / prec
                        
                        if weighting != 'no_weights':
                            
                            mu[:, j] = var[:, j] * tf.reduce_sum(weight_matrix * prec_s * mu_s_c, axis=0)
                        
                        else:
                            mu[:, j] = var[:, j] * tf.reduce_sum(prec_s * mu_s_c, axis=0)
                
                    else:
                        
                        weight_matrix = normalize_weights(weight_matrix)
                        
                        mu[:, j] = tf.reduce_sum(weight_matrix * mu_s_c, axis=0)
                        var[:, j] = tf.reduce_sum(weight_matrix * var_s_c, axis=0)
                

                    
                        
                return self.lik_aggregation(mu, var)


        def lik_aggregation(self, mu, var):

            """Computes expection by numerical intergartion given likelihood and latent gp mean and variances"""

            return self.likelihood.predict_mean_and_var(mu, var)[0]

    return ClassificationModel


def expert_models(name, task):
    if 'expert' in name:
        points_per_experts = int(name.split('_')[1])

        partitioning = name.split('_')[2]

        return classification_model(points_per_experts, partitioning)

