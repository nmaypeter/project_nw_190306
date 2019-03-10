from random import choice
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def get_quantiles(pd, mu, sigma):

    discrim = -2 * sigma**2 * np.log(pd * sigma * np.sqrt(2 * np.pi))

    # no real roots
    if discrim < 0:
        return None

    # one root, where x == mu
    elif discrim == 0:
        return mu

    # two roots
    else:
        return choice([mu - np.sqrt(discrim), mu + np.sqrt(discrim)])


if __name__ == '__main__':
    price_list_g = [[0.24, 0.48, 0.72]]
    # price_list = [[0.24, 0.48, 0.72], [0.24, 0.48, 0.6], [0.24, 0.48, 0.96]]

    for pk in range(len(price_list_g)):
        mu_g = np.mean(price_list_g[pk])
        # mu_g = sum(price_list[pk])
        sigma_g = (max(price_list_g[pk]) - mu_g) / 0.8415
        # sigma_g = 1
        X_g = np.arange(0, 2, 0.001)

        Y_g = [stats.norm.pdf(X_g, mu_g, sigma_g), stats.norm.cdf(X_g, mu_g, sigma_g), stats.norm.sf(X_g, mu_g, sigma_g)]
        X_label = ['wallet guess', 'wallet guess', 'number of nodes with purchasing ability guess']
        Y_label = ['probability density', 'probability', 'probability']
        title = ['pdf', 'cdf', 'ccdf']
        for index in range(len(Y_g)):
            plt.plot(X_g, Y_g[index])
            plt.xlabel(X_label[index])
            plt.ylabel(Y_label[index])
            plt.title(title[index] + ' of normal distribution: μ = ' + str(mu_g) + ', σ = ' + str(round(sigma_g, 4)))
            plt.grid()
            plt.show()

        pw_list = [round(float(Y_g[2][np.argwhere(X_g == p)]), 4) for p in price_list_g[pk]]
        print(pw_list)
        print(round(float(stats.norm.sf(price_list_g[pk][0] + price_list_g[pk][1], mu_g, sigma_g)), 4))
        print(round(float(stats.norm.sf(price_list_g[pk][0] + price_list_g[pk][2], mu_g, sigma_g)), 4))
        print(round(float(stats.norm.sf(price_list_g[pk][1] + price_list_g[pk][2], mu_g, sigma_g)), 4))
        print(round(float(stats.norm.sf(sum(price_list_g[pk]), mu_g, sigma_g)), 4))