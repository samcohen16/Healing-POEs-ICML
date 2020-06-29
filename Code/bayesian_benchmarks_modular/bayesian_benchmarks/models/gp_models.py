import gpflow

def regression_model():
    class SKLWrapperRegression:
        def __init__(self, is_test=False, seed=0):
            pass

        def fit(self, X, Y):
            k = gpflow.kernels.RBF()
            self.model = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
            
            self.opt = gpflow.optimizers.Scipy()
            
            
            
            self.optimize()
        
        def optimize(self):
            
            def objective_closure():
                loss= - self.model.log_marginal_likelihood()
                return loss
            
            opt_logs = self.opt.minimize(objective_closure,
                                    self.model.trainable_variables,
                                    options=dict(maxiter=100))


        def predict(self, Xs):
            predictions = self.model.predict_y(Xs)
            return predictions

        

    return SKLWrapperRegression




def gp_models(name, task):
    if name=='gp' and task=='regression':

        return regression_model()