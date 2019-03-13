from Diffusion_NormalIC import *


class SeedSelectionHDPW:
    def __init__(self, g_dict, s_c_dict, prod_list, total_bud, dis):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### total_bud: (int) the budget to select seed
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### dis: (int) wallet distribution
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.total_budget = total_bud
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
            sigma = 1
        X = np.arange(0, 2, 0.001)
        Y = stats.norm.sf(X, mu, sigma)
        product_weight_list = [round(float(Y[np.argwhere(X == p)]), 4) for p in price_list]

        return product_weight_list

    def constructDegreeDict(self, data_name, p_w_list):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        with open(IniGraph(data_name).data_degree_path) as f:
            for line in f:
                (i, deg) = line.split()
                if deg == '0':
                    continue
                for k in range(self.num_product):
                    deg = str(round(float(deg) * p_w_list[k]))
                    if deg in d_dict:
                        d_dict[deg].add((k, i))
                    else:
                        d_dict[deg] = {(k, i)}
        f.close()

        return d_dict

    def constructExpendDegreeDict(self, p_w_list):
        # -- display the degree and the nodes with the degree --
        ### d_dict: (dict) the degree and the nodes with the degree
        ### d_dict[deg]: (set) the set for deg-degree nodes
        d_dict = {}
        for i in self.graph_dict:
            i_set = {i}
            for i_neighbor in self.graph_dict[i]:
                if i_neighbor not in i_set:
                    i_set.add(i_neighbor)
            for i_neighbor in self.graph_dict[i]:
                for i_neighbor_neighbor in self.graph_dict[i_neighbor]:
                    if i_neighbor_neighbor not in i_set:
                        i_set.add(i_neighbor_neighbor)

            for k in range(self.num_product):
                deg = str(round(len(i_set) * p_w_list[k]))
                if deg in d_dict:
                    d_dict[deg].add((k, i))
                else:
                    d_dict[deg] = {(k, i)}

        return d_dict

    def getHighDegreeNode(self, d_dict, cur_bud):
        # -- get the node with highest degree --
        mep = [-1, '-1']
        max_degree = -1
        while mep[1] == '-1':
            while max_degree == -1:
                for deg in list(d_dict.keys()):
                    if int(deg) > max_degree:
                        max_degree = int(deg)

                if max_degree == -1:
                    return mep, d_dict

                if d_dict[str(max_degree)] == set():
                    del d_dict[str(max_degree)]
                    max_degree = -1

            if d_dict[str(max_degree)] == set():
                del d_dict[str(max_degree)]
                max_degree = -1
                continue

            mep = choice(list(d_dict[str(max_degree)]))
            d_dict[str(max_degree)].remove(mep)
            mep = list(mep)

            if self.seed_cost_dict[mep[1]] + cur_bud > self.total_budget:
                mep[1] = '-1'

        return mep, d_dict


if __name__ == '__main__':
    data_set_name = 'email_undirected'
    product_name = 'r1p3n1'
    total_budget = 10
    distribution_type = 1
    whether_passing_information_without_purchasing = bool(0)
    pp_strategy = 1
    eva_monte_carlo = 1000

    iniG = IniGraph(data_set_name)
    iniP = IniProduct(product_name)

    seed_cost_dict = iniG.constructSeedCostDict()
    graph_dict = iniG.constructGraphDict()
    product_list = iniP.getProductList()
    num_node = len(seed_cost_dict)
    num_product = len(product_list)

    # -- initialization for each budget --
    start_time = time.time()
    sshdpw = SeedSelectionHDPW(graph_dict, seed_cost_dict, product_list, total_budget, distribution_type)
    pw_list = sshdpw.getProductWeight()

    # -- initialization for each sample_number --
    now_budget = 0.0
    seed_set = [set() for _ in range(num_product)]

    degree_dict = sshdpw.constructDegreeDict(data_set_name, pw_list)
    mep_g, degree_dict = sshdpw.getHighDegreeNode(degree_dict, now_budget)
    mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    # -- main --
    while now_budget < total_budget and mep_i_node != '-1':
        seed_set[mep_k_prod].add(mep_i_node)
        now_budget += seed_cost_dict[mep_i_node]

        mep_g, degree_dict = sshdpw.getHighDegreeNode(degree_dict, now_budget)
        mep_k_prod, mep_i_node = mep_g[0], mep_g[1]

    eva = Evaluation(graph_dict, seed_cost_dict, product_list, pp_strategy, whether_passing_information_without_purchasing)
    iniW = IniWallet(data_set_name, product_name, distribution_type)
    wallet_list = iniW.getWalletList()
    personal_prob_list = eva.setPersonalProbList(wallet_list)

    sample_pro_acc, sample_bud_acc = 0.0, 0.0
    sample_sn_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
    sample_pro_k_acc, sample_bud_k_acc = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

    for _ in range(eva_monte_carlo):
        pro, pro_k_list, pnn_k_list = eva.getSeedSetProfit(seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
        sample_pro_acc += pro
        for kk in range(num_product):
            sample_pro_k_acc[kk] += pro_k_list[kk]
            sample_pnn_k_acc[kk] += pnn_k_list[kk]
    sample_pro_acc = round(sample_pro_acc / eva_monte_carlo, 4)
    for kk in range(num_product):
        sample_pro_k_acc[kk] = round(sample_pro_k_acc[kk] / eva_monte_carlo, 4)
        sample_pnn_k_acc[kk] = round(sample_pnn_k_acc[kk] / eva_monte_carlo, 2)
        sample_sn_k_acc[kk] = len(seed_set[kk])
        for sample_seed in seed_set[kk]:
            sample_bud_acc += seed_cost_dict[sample_seed]
            sample_bud_k_acc[kk] += seed_cost_dict[sample_seed]

    print('seed set: ' + str(seed_set))
    print('profit: ' + str(sample_pro_acc))
    print('budget: ' + str(sample_bud_acc))
    print('seed number: ' + str(sample_sn_k_acc))
    print('purchasing node number: ' + str(sample_pnn_k_acc))
    print('ratio profit: ' + str(sample_pro_k_acc))
    print('ratio budget: ' + str(sample_bud_k_acc))
    print('total time: ' + str(round(time.time() - start_time, 2)) + 'sec')