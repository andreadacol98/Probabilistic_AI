import numpy as np
from tqdm import tqdm

# Constants for the Cost function
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.04


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted)**2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted>=true,mask)
    mask_w2 = np.logical_and(np.logical_and(predicted<true,predicted >=THRESHOLD),mask)
    mask_w3 = np.logical_and(predicted<THRESHOLD,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    cost[mask_w3] = cost[mask_w3]*W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted>true,mask)
    mask_w2 = np.logical_and(predicted<=true,mask)

    cost[mask_w1] = cost[mask_w1]*W1
    cost[mask_w2] = cost[mask_w2]*W2
    reward = W4*np.logical_and(predicted < THRESHOLD,true<THRESHOLD)
    if reward is None:
        reward = 0
    return np.mean(cost) - np.mean(reward)


class Model:
    sigma_squared = .0025
    h = .32

    def __init__(self):
        pass

    def mu(self, x):
        return 0.3582

    def k(self, x1, x2):
        return np.exp(-((np.linalg.norm(x1 - x2) / self.h) ** 2))

    def mu_post(self, x):
        n = self.A.shape[0]
        k_xA = np.zeros(n)
        for i in range(0, n):
            k_xA[i] = self.k(x, self.A[i, :])

        return self.mu(x) + k_xA.dot(self.B_dot_delta)

    def k_post(self, x1, x2):
        n = self.A.shape[0]
        k_x1A = np.zeros(n)
        for i in range(0, n):
            k_x1A[i] = self.k(x1, self.A[i, :])
        k_x2A = np.zeros(n)
        for i in range(0, n):
            k_x2A[i] = self.k(x2, self.A[i, :])

        return self.k(x1, x2) - (k_x1A.transpose()).dot(self.B.dot(k_x2A))

    def compute_B(self, train_x):
        print("Deterministic method")
        n = train_x.shape[0]
        K = np.zeros((n, n))
        for col in tqdm(range(0, n)):
            for row in range(0, n):
                K[row, col] = self.k(train_x[row], train_x[col])

        self.B = np.linalg.inv(K + self.sigma_squared*np.eye(n))

    def fit_model(self, train_x, train_y):
        train_x, train_y = small_data(train_x, train_y, 500)
        n = train_x.shape[0]
        self.compute_B(train_x)
        print("Computing mu_A (matrix inversion)")

        # mu_A computation
        mu_A = np.zeros(n)
        for row in range(1, n):
            mu_A[row] = self.mu(train_x[row, :])

        # Store A and delta
        self.A = train_x
        self.B_dot_delta = self.B.dot(train_y - mu_A)

    def scale_minimize_loss(self, ys, vars):
        m = ys.shape[0]
        for i in range(0, m):
            var = (vars[i] + self.sigma_squared)
            if ys[i] + var * 7.57 > .5:
                ys[i] = ys[i] + np.sqrt(var) * 1.1603
            else:
                ys[i] = ys[i] + var * 7.57

            if ys[i] > .6:
                ys[i] = .6
            if 0.465 < ys[i] < .525:
                ys[i] = .49999
        return ys

    def predict(self, test_x):
        m = test_x.shape[0]
        test_y = np.zeros(m)
        vars_y = np.zeros(m)
        for i in range(0, m):
            test_y[i] = self.mu_post(test_x[i, :])
        for i in range(0, m):
            vars_y[i] = self.k_post(test_x[i, :], test_x[i, :])

        print("mu")
        print(test_y)
        print("sigmas")
        print(vars_y)
        return self.scale_minimize_loss(test_y, vars_y)

def small_data(train_x, train_y, req_size):
    n = train_x.shape[0]
    train_x_small = train_x[(n-req_size):n, :]
    train_y_small = train_y[(n-req_size):n]
    return train_x_small, train_y_small

def draw(x, y, test_x, prediction):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x1 = x[:, 0]
    x2 = x[:, 1]
    xt1 = test_x[:, 0]
    xt2 = test_x[:, 1]
    ax.scatter(x1, x2, y, s=1)
    ax.scatter(xt1, xt2, prediction, s=1, c="red")

    plt.show()

def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')

    M = Model()
    M.fit_model(train_x, train_y)
    prediction = M.predict(test_x)

    #train_x_small, train_y_small = small_data(train_x, train_y, 1000)
    #draw(train_x_small, train_y_small, test_x, prediction)


if __name__ == "__main__":
    main()
