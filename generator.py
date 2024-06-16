import numpy as np

class Generator:
    def __init__(self):
        pass

class GridGenerator(Generator):
    def __init__(self):
        super().__init__()

    def generate_checkerboard(self, d):
        if d % 2 == 0:
            v = np.zeros((d, d))
            for i in range(d):
                v[i, i % 2 + np.arange(0, d, 2)] = 1
            return v.flatten()
        else:
            v = np.zeros(d * d)
            v[np.arange(0, d * d, 2)] = 1
            return v

    def generate_strips(self, d):
        if d % 2 == 1:
            v = np.zeros((d, d))
            for i in range(d):
                v[i, i % 2 + np.arange(0, d, 2)] = 1
            return v.flatten()
        else:
            v = np.zeros(d * d)
            v[np.arange(0, d * d, 2)] = 1
            return v

    def generate_halves(self, d):
        v = np.zeros((d, d))
        k = d // 2
        v[:k] = 1
        return v.flatten()

    def generate_halves_purturbed(self, d, q=0.1):
        v = np.zeros((d, d))
        k = d // 2
        v[:k] = 1
        v = v.flatten()
        ridx = np.random.choice(v.shape[0], int(q * v.shape[0]), replace=False)
        v[ridx] = 1
        return v

    @staticmethod
    def generate_multicat_permutation_configuration(d, cs, ns):
        v = np.zeros((d * d))
        cidx = np.arange(d * d)
        for i, (c, n) in enumerate(zip(cs, ns)):
            idx = np.random.choice(cidx.shape[0], n, replace=False)
            v[cidx[idx]] = int(c)
            cidx = np.delete(cidx, idx)
        return v

    @staticmethod
    def perturb_weight(w_map, r=0.1):
        N = w_map.shape[0] * w_map.shape[1]
        W = int(np.sum(w_map) * r / 2)
        ups = np.random.choice(N, W, replace=False)
        downs = np.random.choice(N, W, replace=False)

        w_perturbed = np.zeros_like(w_map.flatten()) + w_map.flatten()
        w_perturbed[ups] += 1
        w_perturbed[downs] -= 1
        w_perturbed[w_perturbed == 2] = 1
        w_perturbed[w_perturbed == -1] = 0

        return w_perturbed.reshape(w_map.shape)

    @staticmethod
    def offset_weight(w_map, r=0.1):
        W = int(np.sum(w_map) * np.abs(r))

        if r < 0:
            idx = np.where(w_map.flatten() == 1)[0]
        else:
            idx = np.where(w_map.flatten() == 0)[0]

        changes = np.random.choice(idx, W, replace=False)

        w_perturbed = np.zeros_like(w_map.flatten()) + w_map.flatten()
        w_perturbed[changes] += np.sign(r)
        # w_perturbed[w_perturbed == 2] = 1
        # w_perturbed[w_perturbed == -1] = 0

        return w_perturbed.reshape(w_map.shape)