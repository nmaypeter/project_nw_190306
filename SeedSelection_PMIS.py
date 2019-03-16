from Diffusion_NormalIC import *
import operator


class SeedSelectionPMIS:
    def __init__(self, g_dict, s_c_dict, prod_list, monte):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### monte: (int) monte carlo times
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.monte = monte

    def generateCelfSequence(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) [k_prod, i_node, mg, flag]
        celf_seq = [[[-1, '-1', 0.0, 0]] for _ in range(self.num_product)]

        diff_ss = Diffusion(self.graph_dict, self.seed_cost_dict, self.product_list, self.monte)

        for i in set(self.graph_dict.keys()):
            s_set = [set() for _ in range(self.num_product)]
            s_set[0].add(i)
            ep = 0.0
            for _ in range(self.monte):
                ep += diff_ss.getSeedSetProfit(s_set)
            ep = round(ep / self.monte, 4)
            mg = round(ep, 4)

            if mg <= 0:
                continue
            for k in range(self.num_product):
                mg = round(mg * self.product_list[k][0] / self.product_list[0][0], 4)
                celf_ep = [k, i, mg, 0]
                celf_seq[k].append(celf_ep)
                for celf_item in celf_seq[k]:
                    if celf_ep[2] >= celf_item[2]:
                        celf_seq[k].insert(celf_seq[k].index(celf_item), celf_ep)
                        celf_seq[k].pop()
                        break

        return celf_seq


if __name__ == '__main__':
    data_set_name = 'email_undirected'
    product_name = 'r1p3n1'
    total_budget = 10
    distribution_type = 1
    whether_passing_information_without_purchasing = bool(0)
    pp_strategy = 1
    monte_carlo, eva_monte_carlo = 100, 1000

    iniG = IniGraph(data_set_name)
    iniP = IniProduct(product_name)

    seed_cost_dict = iniG.constructSeedCostDict()
    graph_dict = iniG.constructGraphDict()
    product_list = iniP.getProductList()
    num_node = len(seed_cost_dict)
    num_product = len(product_list)

    # -- initialization for each budget --
    start_time = time.time()
    sspmis = SeedSelectionPMIS(graph_dict, seed_cost_dict, product_list, monte_carlo)
    diff = Diffusion(graph_dict, seed_cost_dict, product_list, monte_carlo)

    # -- initialization for each sample_number --
    s_matrix, c_matrix = [[] for _ in range(num_product)], [[] for _ in range(num_product)]
    celf_sequence = sspmis.generateCelfSequence()

    for kk in range(num_product):
        s_matrix[kk].append([set() for _ in range(num_product)])
        c_matrix[kk].append(0.0)

        cur_budget, cur_profit = 0.0, 0.0
        seed_set_t = [set() for _ in range(num_product)]

        mep = celf_sequence[kk].pop(0)
        mep_k_prod, mep_i_node, mep_flag = mep[0], mep[1], mep[3]

        while cur_budget < total_budget and mep_i_node != '-1':
            if cur_budget + seed_cost_dict[mep_i_node] > total_budget:
                mep = celf_sequence[kk].pop(0)
                mep_k_prod, mep_i_node, mep_flag = mep[0], mep[1], mep[3]
                if mep_i_node == '-1':
                    break
                continue

            seed_set_length = sum(len(seed_set_t[k]) for k in range(num_product))
            if mep_flag == seed_set_length:
                seed_set_t[mep_k_prod].add(mep_i_node)
                ep_g = 0.0
                for _ in range(monte_carlo):
                    ep_g += diff.getSeedSetProfit(seed_set_t)
                cur_profit = round(ep_g / monte_carlo, 4)
                cur_budget = round(cur_budget + seed_cost_dict[mep_i_node], 2)
                s_matrix[kk].append(copy.deepcopy(seed_set_t))
                c_matrix[kk].append(round(cur_budget, 2))
            else:
                ep1_g = 0.0
                for _ in range(monte_carlo):
                    ep1_g += diff.getExpectedProfit(mep_k_prod, mep_i_node, seed_set_t)
                ep1_g = round(ep1_g / monte_carlo, 4)
                mep_mg = round(ep1_g - cur_profit, 4)
                mep_flag = seed_set_length

                if mep_mg <= 0:
                    continue
                celf_ep_g = [mep_k_prod, mep_i_node, mep_mg, mep_flag]
                celf_sequence[kk].append(celf_ep_g)
                for celf_item_g in celf_sequence[kk]:
                    if celf_ep_g[2] >= celf_item_g[2]:
                        celf_sequence[kk].insert(celf_sequence[kk].index(celf_item_g), celf_ep_g)
                        celf_sequence[kk].pop()
                        break

            mep = celf_sequence[kk].pop(0)
            mep_k_prod, mep_i_node, mep_flag = mep[0], mep[1], mep[3]

    mep_result = [0.0, [set() for _ in range(num_product)]]
    ### bud_index: (list) the using budget index for products
    ### bud_bound_index: (list) the bound budget index for products
    bud_index, bud_bound_index = [len(kk) - 1 for kk in c_matrix], [0 for _ in range(num_product)]
    ### temp_bound_index: (list) the bound to exclude the impossible budget combination s.t. the k-budget is smaller than the temp bound
    temp_bound_index = [0 for _ in range(num_product)]

    while not operator.eq(bud_index, bud_bound_index):
        ### bud_pmis: (float) the budget in this pmis execution
        bud_pmis = 0.0
        for kk in range(num_product):
            bud_pmis += copy.deepcopy(c_matrix)[kk][bud_index[kk]]

        if bud_pmis <= total_budget:
            temp_bound_index = copy.deepcopy(bud_index)
            # -- pmis execution --
            seed_set = [set() for _ in range(num_product)]
            for kk in range(num_product):
                seed_set[kk] = copy.deepcopy(s_matrix)[kk][bud_index[kk]][kk]

            pro_acc = 0.0
            for _ in range(monte_carlo):
                pro_acc += diff.getSeedSetProfit(seed_set)
            pro_acc = round(pro_acc / monte_carlo, 4)

            if pro_acc > mep_result[0]:
                mep_result = [pro_acc, seed_set]

        pointer = num_product - 1
        while bud_index[pointer] == bud_bound_index[pointer]:
            bud_index[pointer] = len(c_matrix[pointer]) - 1
            pointer -= 1
        bud_index[pointer] -= 1
    seed_set = mep_result[1]

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
            sample_bud_acc = round(sample_bud_acc, 2)
            sample_bud_k_acc[kk] = round(sample_bud_k_acc[kk], 2)

    print('seed set: ' + str(seed_set))
    print('profit: ' + str(sample_pro_acc))
    print('budget: ' + str(sample_bud_acc))
    print('seed number: ' + str(sample_sn_k_acc))
    print('purchasing node number: ' + str(sample_pnn_k_acc))
    print('ratio profit: ' + str(sample_pro_k_acc))
    print('ratio budget: ' + str(sample_bud_k_acc))
    print('total time: ' + str(round(time.time() - start_time, 2)) + 'sec')