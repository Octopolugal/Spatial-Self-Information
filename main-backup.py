import json

from libpysal.weights import lat2W
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import norm

from utils import construct_edge_weight, count_multicat_square_switches, evaluate_goodness_of_fit, evaluate_standard_error
from generator import GridGenerator
from surprisal import EmpiricalSurprisal, AnalyticalSurprisal
from plotter import plot_integrated_distribution, plot_probabilities

def integrated_experiment(d, cs, ns, ignores):
    M = cs.shape[0]
    Zs, cat_edge_dicts, scovs, probs = [], dict(zip([(a,b) for a in range(M) for b in range(M)], [[] for _ in range(M*M)])), [], []

    w = lat2W(d, d)

    w_map = construct_edge_weight(w, d)

    analytical_surprisal = AnalyticalSurprisal()
    analytical_surprisal.fit(cs, ns, w_map, ignores)

    r = 1

    N, W = np.sum(ns), np.sum(w_map)
    xmean = np.sum(cs * ns) / np.sum(ns)
    xvar = np.sum((ns * (cs - xmean))**2)

    scaling_factor = N / (W * xvar)

    for i in range(10000):
        Z = GridGenerator.generate_multicat_permutation_configuration(d, cs, ns).reshape((d,d))
        X = Z.flatten().reshape((1,-1)) - np.mean(Z)
        Y = np.matmul(w_map, X.T)

        Zs.append(Z)
        scovs.append(np.matmul(X, Y).flatten())
        # probs.append(analytical_surprisal.get_probability(Z, w_map))
        qdict = count_multicat_square_switches(Z, cs)

        for k in cat_edge_dicts.keys():
            cat_edge_dicts[k].append(r * qdict[k])

    Zs = np.array(Zs)
    scovs = np.array(scovs).flatten()
    moran_Is = scovs * scaling_factor

    mus, sigmas, ds, ijs = {}, {}, [], []
    for k in cat_edge_dicts.keys():
        (mu, sigma) = norm.fit(cat_edge_dicts[k])

        mus[k] = mu
        sigmas[k] = sigma
        ds.append((cs[k[0]] - xmean) * (cs[k[1]] - xmean))
        ijs.append(ignores[k[0]] * ignores[k[1]])

        # plt.figure()
        # n, bins, patches = plt.hist(cat_edge_dicts[k], 60, density=True, facecolor='green', alpha=0.75)

        # add a 'best fit' line
        # y = norm.pdf(bins, mu, sigma)
        # l = plt.plot(bins, y, 'r--', linewidth=2)
        #
        # plt.xlabel('Smarts')
        # plt.ylabel('Probability')
        # plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=%.3f,\ \sigma=%.3f$' %(mu, sigma))
        # plt.grid(True)
        # plt.savefig("figures/{}.png".format(k))
        #
        # plt.show()

    # Mu_empirical = EmpiricalSurprisal.compute_mean(cs, ns, mus)
    # Sigma_empirical = EmpiricalSurprisal.compute_std(cs, ns, sigmas, ignores)
    Mu_empirical = np.mean(scovs)
    Sigma_empirical = np.std(scovs)

    Mu_analytical, Mu_dict, Mu_coef_dict = AnalyticalSurprisal.compute_mean(cs, ns)
    Sigma_analytical, Sigma_dict, Sigma_coef_dict = AnalyticalSurprisal.compute_std(cs, ns, ignores=ignores)

    (Mu_gt, Sigma_gt) = norm.fit(scovs)

    # print(Mu_empirical, Sigma_empirical, Mu_analytical, Sigma_analytical, Mu_gt, Sigma_gt)

    return Zs, scovs, scaling_factor, (Mu_empirical, Sigma_empirical, Mu_analytical, Sigma_analytical, Mu_gt, Sigma_gt), Mu_dict, Mu_coef_dict, Sigma_dict, Sigma_coef_dict, mus, sigmas


if __name__ == '__main__':
    expr_name = "0"
    K = 4
    d, cs, ignores = 40, np.array([0, 1, 2, 3]), np.array([0, 1, 1, 1])
    params = {
        "low": np.array([d * d - 1200, 400, 400, 400]),
        "medium": np.array([d * d - 900, 300, 300, 300]),
        "high": np.array([d * d - 600, 200, 200, 200])
    }

    results = {}
    beta1 = 1 - (4 * d) / (K * d * d)

    for label, ns in params.items():
        # w = lat2W(d, d)
        # w_map = construct_edge_weight(w, d)
        #
        # Z = GridGenerator.generate_multicat_permutation_configuration(d, cs, ns).reshape((d, d))
        # print(compute_probability(Z, cs, ns, w_map, ignores))

        Zs, scovs, scaling_factor, (Mu_empirical, Sigma_empirical, Mu_analytical, Sigma_analytical, Mu_gt, Sigma_gt), Mu_dict, Mu_coef_dict, Sigma_dict, Sigma_coef_dict, mu_dict, sigma_dict = integrated_experiment(d, cs, ns, ignores)

        Mu_corrected = 0.
        for k in Mu_dict.keys():
            print(k, mu_dict[k] / Mu_dict[k])
            if k[0] != 0 and k[1] != 0 and k[0] == k[1]:
                beta2 = 1 - 2 / ns[k[0]]
                Mu_corrected += beta1 * Mu_coef_dict[k] * Mu_dict[k] / beta2
            else:
                Mu_corrected += beta1 * Mu_coef_dict[k] * Mu_dict[k]

        print(Mu_corrected, Mu_gt, Sigma_analytical, Sigma_gt)

        # Mu_estimated = 0.
        # for k in Mu_dict.keys():
        #     print(k, Mu_dict[k], mu_dict[k], Sigma_dict[k], sigma_dict[k])
        #     Mu_estimated += Mu_coef_dict[k] * mu_dict[k]
        #
        # print(Mu_analytical, Mu_gt, Mu_estimated)
        #
        # print(Mu_dict)
        # print(Mu_coef_dict)
        # print(mu_dict)

        # json.dump({"analytical_mus": Mu_dict, "analytical_coef": Mu_coef_dict, "estimated_mus": mu_dict}, open("estimated_dicts.json", "w"))
        # print(Mu_analytical, Mu_gt, Sigma_analytical, Sigma_gt)

        # from matplotlib import pyplot as plt
        # plt.hist((scovs - Mu_analytical) / Sigma_analytical, bins=60)
        # plt.hist((scovs - Mu_gt) / Sigma_gt, bins=60)
        # plt.savefig("figures/standardized.png")

        print(evaluate_goodness_of_fit(scovs, Mu_corrected, Sigma_analytical))
        # print(evaluate_goodness_of_fit(scovs, Mu_analytical, Sigma_gt))
        # print(evaluate_goodness_of_fit(scovs, Mu_gt, Sigma_analytical))
        # print(evaluate_goodness_of_fit(scovs, Mu_gt, Sigma_gt))
        # print(evaluate_standard_error(np.sum(ns), Mu_analytical, Sigma_analytical, Mu_gt, Sigma_gt))

        # plot_probabilities(scovs * scaling_factor, probs)
        #
        #
        # results[label] = [Zs, scovs, scaling_factor, (Mu_empirical, Sigma_empirical, Mu_analytical, Sigma_analytical, Mu_gt, Sigma_gt)]
        #
        # mu_diff, sigma_diff = np.mean(scovs) - Mu_analytical, np.abs(Sigma_analytical - Sigma_gt)
        #
        # print(label, (np.sqrt(scovs.shape[0]) * (np.mean(scovs) - Mu_analytical)) / np.std(scovs), (scovs.shape[0] - 1) * np.var(scovs) / Sigma_analytical**2)
        #
        plot_integrated_distribution(label, scovs, 1, Mu_corrected, Sigma_analytical, Mu_gt, Sigma_gt)

    # json.dump(results, open('results/{}.json'.format(expr_name), "w"))