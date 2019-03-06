import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

price_list = [[0.24, 0.48, 0.72]]
# price_list = [[0.24, 0.48, 0.72], [0.24, 0.48, 0.6], [0.24, 0.48, 0.96]]

for pk in range(len(price_list)):
    mu = np.median(price_list[pk])
    # mu = np.mean(price_list[pk])
    # mu = sum(price_list[pk])
    # sigma = round(float(np.std(price_list[pk])), 4)
    sigma = (max(price_list[pk]) - mu) / 0.8415
    # sigma = 1
    X = np.arange(0, 2, 0.001)

    Y = [stats.norm.pdf(X, mu, sigma), stats.norm.cdf(X, mu, sigma), stats.norm.sf(X, mu, sigma)]
    X_label = ['wallet guess', 'wallet guess', 'number of nodes with purchasing ability guess']
    Y_label = 'probability'
    title = ['pdf', 'cdf', 'ccdf']
    for index in range(len(Y)):
        plt.plot(X, Y[index])
        plt.xlabel(X_label[index])
        plt.ylabel(Y_label)
        plt.title(title[index] + ' of normal distribution: μ = ' + str(mu) + ', σ = ' + str(round(sigma, 4)))
        plt.grid()
        plt.show()

    pw_list = [round(float(Y[2][np.argwhere(X == p)]), 4) for p in price_list[pk]]
    print(pw_list)
    print(round(float(stats.norm.sf(price_list[pk][0] + price_list[pk][1], mu, sigma)), 4))
    print(round(float(stats.norm.sf(price_list[pk][0] + price_list[pk][2], mu, sigma)), 4))
    print(round(float(stats.norm.sf(price_list[pk][1] + price_list[pk][2], mu, sigma)), 4))
    print(round(float(stats.norm.sf(sum(price_list[pk]), mu, sigma)), 4))