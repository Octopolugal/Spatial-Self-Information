from matplotlib import pyplot as plt

import numpy as np
from scipy.stats import norm

def plot_integrated_distribution(label, scovs, scaling_factor, Mu_analytical, Sigma_analytical, Mu_gt, Sigma_gt):
    # mu_error, sigma_error = np.abs((Mu_analytical - Mu_gt) / Mu_gt), (np.abs(Sigma_analytical - Sigma_gt) / Sigma_gt) ** 2

    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'text.usetex': False, 'font.family': 'Helvetica'})
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    n, bins, patches = plt.hist(scovs * scaling_factor, 60, density=True, facecolor='green', alpha=0.3)

    # add a 'best fit' line
    # y = norm.pdf(bins, Mu_empirical, Sigma_empirical)
    # plt.plot(bins, y, 'r--', linewidth=2, label="empirical")

    y = norm.pdf(bins, Mu_analytical  * scaling_factor, Sigma_analytical * scaling_factor)
    plt.plot(bins, y, 'r--', linewidth=5, label="analytical")

    y = norm.pdf(bins, Mu_gt  * scaling_factor, Sigma_gt  * scaling_factor)
    plt.plot(bins, y, 'b--', linewidth=5, label="estimated")

    plt.xlabel(r"$\bar{I}$, Unscaled Moran's I", fontsize=40, labelpad=20)
    plt.ylabel('Probability Density', fontsize=40, labelpad=20)
    plt.tight_layout()
    # plt.title("Error in ", fontsize=10)
    plt.grid(True)

    plt.legend(fontsize=30)

    plt.savefig("figures/integrated-approximation-{}-independence.png".format(label))

def plot_probabilities(moran_Is, probs):
    plt.scatter(moran_Is, probs)
    plt.savefig("figures/moran-I-probabilities.png")

