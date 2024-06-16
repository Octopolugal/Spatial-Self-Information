import json

from libpysal.weights import lat2W
import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import norm

from utils import construct_edge_weight, count_multicat_square_switches, evaluate_goodness_of_fit, evaluate_standard_error, evaluate_kl_divergence
from generator import GridGenerator
from surprisal import EmpiricalSurprisal, AnalyticalSurprisal
from plotter import plot_integrated_distribution, plot_probabilities

def compute_probability(Z, cs, ns, w_map, ignores):
    analytical_surprisal = AnalyticalSurprisal()
    analytical_surprisal.fit(cs, ns, w_map, ignores)

    # TO-DO: iterate over image

    moran_I = analytical_surprisal.get_probability(Z, w_map)
    probability = analytical_surprisal.get_probability(Z, w_map)

    return moran_I, probability

def run_an_experiment(d, w_map, cs, ns, ignores):
    M = cs.shape[0]
    Zs, cat_edge_dicts, scovs, probs = [], dict(zip([(a,b) for a in range(M) for b in range(M)], [[] for _ in range(M*M)])), [], []

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
        qdict = count_multicat_square_switches(Z, cs)

        for k in cat_edge_dicts.keys():
            cat_edge_dicts[k].append(r * qdict[k])

    Zs = np.array(Zs)
    scovs = np.array(scovs).flatten()

    mus, sigmas, ds, ijs = {}, {}, [], []
    for k in cat_edge_dicts.keys():
        (mu, sigma) = norm.fit(cat_edge_dicts[k])

        mus[k] = mu
        sigmas[k] = sigma
        ds.append((cs[k[0]] - xmean) * (cs[k[1]] - xmean))
        ijs.append(ignores[k[0]] * ignores[k[1]])

    Mu_analytical, Mu_dict, Mu_coef_dict = AnalyticalSurprisal.compute_mean(cs, ns)
    Sigma_analytical, Sigma_dict, Sigma_coef_dict = AnalyticalSurprisal.compute_std(cs, ns, ignores=ignores)

    (Mu_gt, Sigma_gt) = norm.fit(scovs)

    return {
        "Zs": Zs,
        "scovs": scovs,
        "scaling_factor": scaling_factor,
        "mu_coef_dict": Mu_coef_dict,
        "sigma_coef_dict": Sigma_coef_dict,
        "mu_analytical_dict": Mu_dict,
        "sigma_analytical_dict": Sigma_dict,
        "mu_empirical_dict": mus,
        "sigma_empirical_dict": sigmas,
        "Sigma_analytical": Sigma_analytical,
        "Mu_estimated": Mu_gt,
        "Sigma_estimated": Sigma_gt
    }

def run_a_perturbation_experiment(Zs, w_map, cs, ns, ignores):
    M = cs.shape[0]
    cat_edge_dicts, scovs, probs = dict(zip([(a,b) for a in range(M) for b in range(M)], [[] for _ in range(M*M)])), [], []

    r = 1

    N, W = np.sum(ns), np.sum(w_map)
    xmean = np.sum(cs * ns) / np.sum(ns)
    xvar = np.sum((ns * (cs - xmean))**2)

    scaling_factor = N / (W * xvar)

    for Z in Zs:
        X = Z.flatten().reshape((1,-1)) - np.mean(Z)
        Y = np.matmul(w_map, X.T)

        scovs.append(np.matmul(X, Y).flatten())
        qdict = count_multicat_square_switches(Z, cs)

        for k in cat_edge_dicts.keys():
            cat_edge_dicts[k].append(r * qdict[k])

    scovs = np.array(scovs).flatten()

    mus, sigmas, ds, ijs = {}, {}, [], []
    for k in cat_edge_dicts.keys():
        (mu, sigma) = norm.fit(cat_edge_dicts[k])

        mus[k] = mu
        sigmas[k] = sigma
        ds.append((cs[k[0]] - xmean) * (cs[k[1]] - xmean))
        ijs.append(ignores[k[0]] * ignores[k[1]])

    Mu_analytical, Mu_dict, Mu_coef_dict = AnalyticalSurprisal.compute_mean(cs, ns)
    Sigma_analytical, Sigma_dict, Sigma_coef_dict = AnalyticalSurprisal.compute_std(cs, ns, ignores=ignores)

    (Mu_gt, Sigma_gt) = norm.fit(scovs)

    return {
        "scovs": scovs,
        "scaling_factor": scaling_factor,
        "mu_coef_dict": Mu_coef_dict,
        "sigma_coef_dict": Sigma_coef_dict,
        "mu_analytical_dict": Mu_dict,
        "sigma_analytical_dict": Sigma_dict,
        "mu_empirical_dict": mus,
        "sigma_empirical_dict": sigmas,
        "Sigma_analytical": Sigma_analytical,
        "Mu_estimated": Mu_gt,
        "Sigma_estimated": Sigma_gt
    }

if __name__ == '__main__':
    expr_name = "0"
    d = 50
    # d, cs, ignores = 50, np.array([0, 1, 2, 3]), np.array([0, 1, 1, 1])
    # params = {
    #     "low": np.array([d * d - 1200, 400, 400, 400]),
    #     "medium": np.array([d * d - 900, 300, 300, 300]),
    #     "high": np.array([d * d - 600, 200, 200, 200])
    # }
    w = lat2W(d, d)
    w_map = construct_edge_weight(w, d)

    data = np.load("notebooks/sliced-data-bin-50.npz")["data"]
    analytical_surprisal = AnalyticalSurprisal()

    bg_rates, rmaxes = [], []
    for dt in data:
        cs, ns = np.unique(dt, return_counts=True)
        bg_rates.append(np.max(ns) / (d*d))
        rmaxes.append(np.argmax(ns))

    bg_rates = np.array(bg_rates)
    print(bg_rates)
    rmaxes = np.array(rmaxes)

    # group1_idx = np.where((bg_rates > 0.3) & (bg_rates < 0.45))
    # group2_idx = np.where((bg_rates > 0.45) & (bg_rates < 0.7))

    # print(group1_idx)

    probs, morans, mus, sigmas, scalings = [], [], [], [], []

    for i, dt in enumerate(data):
        cs, ns = np.unique(dt, return_counts=True)
        ignores = np.ones_like(cs)
        ignores[rmaxes[i]] = 0
        analytical_surprisal.fit(cs, ns, w_map, ignores)
        analytical_surprisal.get_moran_I_upper(dt, w_map)

        moran = analytical_surprisal.get_moran_I_upper(dt, w_map)
        prob = analytical_surprisal.get_probability(dt, w_map)
        mu, sigma, scaling = analytical_surprisal.get_fitted_params()

        print(moran, scaling, prob, mu, sigma)

        probs.append(prob)
        morans.append(moran)
        mus.append(mu)
        sigmas.append(sigma)
        scalings.append(scaling)

    np.savez("slope-data-bin-50.npz", bg_rates=bg_rates, morans=morans, probs=probs, mus=mus, sigmas=sigmas, scalings=scalings)

    # results = {}
    # beta1 = 1 - (4 * d) / (K * d * d)
    #
    # data, kls, S = [], [], []
    #
    # for i in np.arange(200, 301, 20):
    #     ns = np.array([d * d - 3*i, i, i, i])
    #     print(ns)
    #     tmp = []
    #     Ss = []
    #     kl = []
    #
    #     # Zs = []
    #     # for _ in range(10000):
    #     #     Z = GridGenerator.generate_multicat_permutation_configuration(d, cs, ns).reshape((d, d))
    #     #     Zs.append(Z)
    #     for j in range(10):
    #         # w_perturbed = GridGenerator.offset_weight(w_map, r=i)
    #         # print("Weight difference: ", np.sum(w_perturbed) - np.sum(w_map))
    #         # result = run_a_perturbation_experiment(Zs, w_perturbed, cs, ns, ignores)
    #         result = run_an_experiment(d, w_map, cs, ns, ignores)
    #         Mu_dict = result['mu_analytical_dict']
    #         mu_dict = result['mu_empirical_dict']
    #         Mu_coef_dict = result['mu_coef_dict']
    #         Mu_estimated = result['Mu_estimated']
    #         Sigma_estimated = result['Sigma_estimated']
    #
    #         Sigma_analytical = result['Sigma_analytical']
    #         Mu_uncorrected = 0.
    #         Mu_corrected = 0.
    #
    #         Stmp = []
    #         kltmp = []
    #         for k in Mu_dict.keys():
    #             print(k, mu_dict[k] / Mu_dict[k])
    #             # if k[0] != 0 and k[1] != 0 and k[0] == k[1]:
    #             if k[0] == k[1]:
    #                 beta2 = 1 - 2 / Mu_dict[k] // important!!!
    #                 Stmp.append([beta1 * Mu_dict[k] / beta2, beta1 * Mu_dict[k], mu_dict[k]])
    #                 Mu_corrected += beta1 * Mu_coef_dict[k] * Mu_dict[k] / beta2
    #             else:
    #                 # S_corrected = beta1 * Mu_dict[k]
    #                 Mu_corrected += beta1 * Mu_coef_dict[k] * Mu_dict[k]
    #
    #             # S_uncorrected = beta1 * Mu_dict[k]
    #             Mu_uncorrected += beta1 * Mu_coef_dict[k] * Mu_dict[k]
    #
    #         tmp.append([Mu_uncorrected, Mu_corrected, Mu_estimated, Sigma_analytical, Sigma_estimated])
    #         Ss.append(Stmp)
    #
    #         # print(Mu_corrected, Sigma_analytical, Mu_estimated, Sigma_estimated)
    #
    #         x = np.linspace(norm.ppf(0.01, loc=Mu_estimated, scale=Sigma_estimated), norm.ppf(0.99, loc=Mu_estimated, scale=Sigma_estimated), 100)
    #         empirical_ys = norm.pdf(x, loc=Mu_estimated, scale=Sigma_estimated)
    #         corrected_ys = norm.pdf(x, loc=Mu_corrected, scale=Sigma_analytical)
    #         uncorrected_ys = norm.pdf(x, loc=Mu_uncorrected, scale=Sigma_analytical)
    #
    #         # print(empirical_ys.shape, analytical_ys.shape)
    #
    #         kl1 = evaluate_kl_divergence(corrected_ys, empirical_ys)
    #         kl2 = evaluate_kl_divergence(uncorrected_ys, empirical_ys)
    #         kl.append([np.sum(kl1), np.sum(kl2)])
    #
    #     print(tmp)
    #     data.append(tmp)
    #     kls.append(kl)
    #     S.append(Ss)
    #
    # # print(kls)
    # np.savez("beta2-correction.npz", data=data, kl=kls, S=S)

    # analytical_surprisal = AnalyticalSurprisal()
    #
    # results = {}
    # beta1 = 1 - (4 * d) / (K * d * d)
    #
    # cs = np.array([0, 1, 2, 3])
    # ns = np.array([d * d - 3*1000, 1000, 1000, 1000])
    # ignores = np.array([0, 1, 1, 1])
    #
    # result = run_an_experiment(d, w_map, cs, ns, ignores)
    # Mu_dict = result['mu_analytical_dict']
    # mu_dict = result['mu_empirical_dict']
    # Mu_coef_dict = result['mu_coef_dict']
    # Mu_estimated = result['Mu_estimated']
    # Sigma_estimated = result['Sigma_estimated']
    # scaling_factor = result['scaling_factor']
    #
    # Sigma_analytical = result['Sigma_analytical']
    # Mu_uncorrected = 0.
    # Mu_corrected = 0.
    #
    # for k in Mu_dict.keys():
    #     # print(k, mu_dict[k] / Mu_dict[k])
    #     # if k[0] != 0 and k[1] != 0 and k[0] == k[1]:
    #     if k[0] == k[1]:
    #         beta2 = 1 - 2 / ns[k[0]]
    #         Mu_corrected += beta1 * Mu_coef_dict[k] * Mu_dict[k] / beta2
    #     else:
    #         Mu_corrected += beta1 * Mu_coef_dict[k] * Mu_dict[k]
    #
    # Z = GridGenerator.generate_multicat_permutation_configuration(d, cs, ns).reshape((d,d))
    # Mu_wiki, Sigma_wiki = analytical_surprisal.compute_wiki_mean_and_std(w_map, Z)
    #
    #
    # print("Mu Wiki: ", Mu_wiki, "Mu corrected: ", Mu_corrected * scaling_factor, "Mu empirical:", Mu_estimated * scaling_factor)
    # print("Sigma Wiki: ", Sigma_wiki, "Sigma corrected: ", Sigma_analytical * scaling_factor, "Sigma empirical:",
    #       Sigma_estimated * scaling_factor)



