from Diffusion_NormalIC import *


class SeedSelectionNGAP:
    def __init__(self, g_dict, s_c_dict, prod_list):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)

    def generateCelfSequence(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) [k_prod, i_node, mg, flag]
        celf_seq = [[-1, '-1', 0.0, 0]]

        diff_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in set(self.graph_dict.keys()):
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = diff_ss.getSeedSetProfit(s_set)
            mg = round(ep, 4)

            if mg <= 0:
                continue
            for k in range(self.num_product):
                mg = round(mg * self.product_list[k][0] / self.product_list[0][0], 4)
                celf_ep = [k, i, mg, 0]
                celf_seq.append(celf_ep)
                for celf_item in celf_seq:
                    if celf_ep[2] >= celf_item[2]:
                        celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                        celf_seq.pop()
                        break

        return celf_seq

    def generateCelfSequenceR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) [k_prod, i_node, mg, flag]
        celf_seq = [[-1, '-1', 0.0, 0]]

        diff_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in set(self.graph_dict.keys()):
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = diff_ss.getSeedSetProfit(s_set)
            mg = round(ep, 4)

            if mg <= 0:
                continue
            for k in range(self.num_product):
                mg = round(mg * self.product_list[k][0] / self.product_list[0][0], 4)
                if self.seed_cost_dict[i] == 0:
                    break
                else:
                    mg_ratio = round(mg / self.seed_cost_dict[i], 4)
                celf_ep = [k, i, mg_ratio, 0]
                celf_seq.append(celf_ep)
                for celf_item in celf_seq:
                    if celf_ep[2] >= celf_item[2]:
                        celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                        celf_seq.pop()
                        break

        return celf_seq


class SeedSelectionNGAPPW:
    def __init__(self, g_dict, s_c_dict, prod_list, dis):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### dis: (int) wallet distribution
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.dis = dis

    def getProductWeight(self):
        price_list = [k[2] for k in self.product_list]
        mu, sigma = 0, 1
        if self.dis == 1:
            mu = np.mean(price_list)
            sigma = (max(price_list) - mu) / 0.8415
        elif self.dis == 2:
            mu = sum(price_list)
            sigma = abs(min(price_list) - mu) / 3
        X = np.arange(0, 2, 0.001)
        Y = stats.norm.sf(X, mu, sigma)
        product_weight_list = [round(float(Y[np.argwhere(X == p)]), 4) for p in price_list]

        return product_weight_list

    def generateCelfSequence(self, pw_list):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) [k_prod, i_node, mg, flag]
        celf_seq = [[-1, '-1', 0.0, 0]]

        diff_ss = DiffusionAccProbPW(self.graph_dict, self.seed_cost_dict, self.product_list, pw_list)

        for i in set(self.graph_dict.keys()):
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = diff_ss.getSeedSetProfit(s_set)
            mg = round(ep, 4)

            if mg <= 0:
                continue
            for k in range(self.num_product):
                mg = round(mg * self.product_list[k][0] / self.product_list[0][0], 4)
                celf_ep = [k, i, mg, 0]
                celf_seq.append(celf_ep)
                for celf_item in celf_seq:
                    if celf_ep[2] >= celf_item[2]:
                        celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                        celf_seq.pop()
                        break

        return celf_seq

    def generateCelfSequenceR(self, pw_list):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) [k_prod, i_node, mg, flag]
        celf_seq = [[-1, '-1', 0.0, 0]]

        diff_ss = DiffusionAccProbPW(self.graph_dict, self.seed_cost_dict, self.product_list, pw_list)

        for i in set(self.graph_dict.keys()):
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = diff_ss.getSeedSetProfit(s_set)
            mg = round(ep, 4)

            if mg <= 0:
                continue
            for k in range(self.num_product):
                mg = round(mg * self.product_list[k][0] / self.product_list[0][0], 4)
                if self.seed_cost_dict[i] == 0:
                    break
                else:
                    mg_ratio = round(mg / self.seed_cost_dict[i], 4)
                celf_ep = [k, i, mg_ratio, 0]
                celf_seq.append(celf_ep)
                for celf_item in celf_seq:
                    if celf_ep[2] >= celf_item[2]:
                        celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                        celf_seq.pop()
                        break

        return celf_seq