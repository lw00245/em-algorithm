import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def p_x_given_y(y, mus, sigmas):
    mu = mus[0] + sigmas[1, 0] / sigmas[0, 0] * (y - mus[1])
    sigma = sigmas[0, 0] - sigmas[1, 0] / sigmas[1, 1] * sigmas[1, 0]
    return np.random.normal(mu, sigma)


def p_y_given_x(x, mus, sigmas):
    mu = mus[1] + sigmas[0, 1] / sigmas[1, 1] * (x - mus[0])
    sigma = sigmas[1, 1] - sigmas[0, 1] / sigmas[0, 0] * sigmas[0, 1]
    return np.random.normal(mu, sigma)


def gibbs_sampling(mus, sigmas, iter=10000):
    samples = np.zeros((iter, 2))
    y = np.random.rand() * 10

    for i in range(iter):
        x = p_x_given_y(y, mus, sigmas)
        y = p_y_given_x(x, mus, sigmas)
        samples[i, :] = [x, y]

    return samples


def hastings_sampler(MN_OBJ,iter=10000):

    samples = np.zeros((iter, 2))
    d_old = np.random.rand(2)
    for i in range(iter):
        delta = np.random.multivariate_normal([0,0],[[1,0.5],[0.5,1]])
        d_new = d_old + delta
        u = np.random.rand()
        samples[i,:] = d_old
        if u <= MN_OBJ.pdf(d_new)/MN_OBJ.pdf(d_old):
            samples[i,:] = d_new
            d_old = d_new
        """
        x = p_x_given_y(y, mus, sigmas)
        y = p_y_given_x(x, mus, sigmas)
        samples[i, :] = [x, y]
        """

    return samples

if __name__ == '__main__':
    mus = np.array([5, 5])
    sigmas = np.array([[1, .9], [.9, 1]])
    from scipy.stats import multivariate_normal as MN
    MN_OBJ = MN(mus,sigmas)
    samples = hastings_sampler(MN_OBJ)
    #samples = gibbs_sampling(mus, sigmas)
    a = sns.jointplot(samples[:, 0], samples[:, 1])
    plt.show()
    a.savefig("./test.png")
    print("test")
