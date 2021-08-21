import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from GP import GP
import matplotlib.pyplot as plt
from tqdm import tqdm

domain = np.array([[0, 5]])

class BO_algo():
    def __init__(self):
        ker_f = .5*Matern(length_scale=.5, nu=2.5)
        ker_v = np.sqrt(2)*Matern(length_scale=.5, nu=2.5)
        self.reg_f = GP(ker=ker_f, noise=0.15)
        self.reg_v = GP(ker=ker_v, noise=0.0001)
        self.epoch = -1
        self.beta = 5

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            1 x domain.shape[0] array containing the next point to evaluate
        """
        if not hasattr(self, 'X'):
            return np.array([[np.random.uniform(0, 5)]])
        return self.optimize_acquisition_function()
        raise NotImplementedError


    def optimize_acquisition_function(self):
        """
        Optimizes the acquisition function.

        Returns
        -------
        x_opt: np.ndarray
            1 x domain.shape[0] array containing the point that maximize the acquisition function.
        """
        x_values = np.linspace(domain[:, 0], domain[:, 1], num=1000)
        f_values = np.zeros(x_values.shape)
        for i, x in enumerate(x_values):
            f_values[i] = self.acquisition_function(x)
        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])


    def acquisition_function(self, x):
        if self.epoch < self.X.shape[0]:
            self.reg_f.fit(self.X, self.f)
            self.epoch = self.X.shape[0]
        mu = self.reg_f.mu_post(x)
        sigma = self.reg_f.k_post(x2=x,x1=x)
        return (mu + self.beta*sigma)

    def mean_function(self, x):
        if self.epoch < self.X.shape[0]:
            self.reg_f.fit(self.X, self.f)
        mu = self.reg_f.mu_post(x)
        return mu

    def add_data_point(self, x, f, v):
        """
        Add data points to the model.

         Parameters
        ----------
        x: np.ndarray
            Hyperparameters
        f: np.ndarray
            Model accuracy
        v: np.ndarray
            Model training speed
        """
        if not hasattr(self, 'X'):
            print("Starting Job")
            self.X = x
        else:
            self.X = np.vstack((self.X,x))
        if not hasattr(self, 'f'):
            self.f = np.array([f])
        else:
            self.f = np.vstack((self.f,np.array([f])))
        if not hasattr(self, 'v'):
            self.v = np.array([[v]])
        else:
            self.v = np.vstack((self.v,np.array([[v]])))

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: np.ndarray
            1 x domain.shape[0] array containing the optimal solution of the problem
        """
        x_values = np.linspace(domain[:, 0], domain[:, 1], num=1000)
        f_values = np.zeros(x_values.shape)
        for i, x in enumerate(x_values):
            f_values[i] = self.mean_function(x)
        ind = np.argmax(f_values)
        return np.atleast_2d(x_values[ind])


""" Toy problem to check code works as expected """

def check_in_domain(x):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= domain[None, :, 0]) and np.all(x <= domain[None, :, 1])

class random_state:
    def __init__(self):
        ker_f = .5*Matern(length_scale=.5, nu=2.5)
        ker_v = np.sqrt(2)*Matern(length_scale=.5, nu=2.5)
        self.reg_f = GaussianProcessRegressor(kernel=ker_f, alpha=0.15)
        self.reg_v = GaussianProcessRegressor(kernel=ker_v, alpha=0.0001)
        self.v_shift = 1.5
        self.xs = np.linspace(domain[0, 0], domain[0, 1], 1000).reshape(-1,1)
        self.fs = self.reg_f.sample_y(self.xs.reshape(-1,1), random_state=None)
        self.vs = self.reg_v.sample_y(self.xs, random_state=None)+self.v_shift

    def f(self, x, noisy = True):
        """Dummy objective"""
        if noisy:
            return np.interp(x, self.xs.flatten(), self.fs.flatten())[0] + np.random.normal(0,.15,1)[0]
        else:
            return np.interp(x, self.xs.flatten(), self.fs.flatten())[0]

    def v(self, x, noisy = True):
        """Dummy speed"""
        if noisy:
            return np.interp(x, self.xs.flatten(), self.vs.flatten())[0] + self.v_shift + np.random.normal(0,.0001,1)[0]
        else:
            return np.interp(x, self.xs.flatten(), self.vs.flatten())[0] + self.v_shift


def main():
    rf = random_state()
    # Init problem
    agent = BO_algo()
    # Loop until budget is exhausted
    xs = []
    os = []
    verbose = 0
    for j in tqdm(range(20)):
        # Get next recommendation
        x = agent.next_recommendation()
        # Check for valid shape
        assert x.shape == (1, domain.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {domain.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = rf.f(x)
        cost_val = rf.v(x)
        xx = np.linspace(domain[0, 0], domain[0, 1], 1000).reshape(-1,1)
        ff = np.zeros(xx.shape)
        sigmas = np.zeros(xx.shape)
        xs.append(x[0, 0])
        os.append(obj_val)
        if j > 2:
            for i, x_ in enumerate(xx):
                ff[i] = agent.reg_f.mu_post(x_)
                sigmas[i] = agent.reg_f.k_post(x2=x_, x1=x_)
            if(verbose == 4):
                plt.close()
                plt.plot(xx,ff)
                plt.plot(xx,ff + agent.beta*sigmas.reshape(ff.shape))
                plt.plot(rf.xs,rf.fs)
                xs.append(x[0,0])
                os.append(obj_val)
                plt.plot(xs, os, '*')
                plt.plot(x, obj_val, '*g')
                plt.show()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = np.atleast_2d(agent.get_solution())
    assert solution.shape == (1, domain.shape[0]), \
        f"The function get solution must return a numpy array of shape (" \
        f"1, {domain.shape[0]})"
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the ' \
        f'domain, {solution} returned instead'

    # Compute regret
    if rf.v(solution, noisy=False) < 1.2:
        regret = 1
    else:
        regret = (max(rf.fs) - rf.f(solution, noisy=False))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{rf.f(solution)}\nRegret{regret}')


if __name__ == "__main__":
    main()
