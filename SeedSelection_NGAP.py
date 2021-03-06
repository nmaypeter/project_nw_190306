from Diffusion import *


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
        ### celf_ep: (list) (k_prod, i_node, mg, flag)
        celf_seq = [(-1, '-1', 0.0, 0)]
        diffap_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeDict({i}, i, 1)
            ei = diffap_ss.getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0], 4)
                    celf_ep = (k, i, mg, 0)
                    celf_seq.append(celf_ep)
                    for celf_item in celf_seq:
                        if celf_ep[2] >= celf_item[2]:
                            celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                            celf_seq.pop()
                            break

        return celf_seq

    def generateCelfSequenceR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) (k_prod, i_node, mg, flag)
        celf_seq = [(-1, '-1', 0.0, 0)]
        diffap_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeDict({i}, i, 1)
            ei = diffap_ss.getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    if self.seed_cost_dict[i] == 0:
                        break
                    else:
                        mg_r = round(ei * self.product_list[k][0] / self.seed_cost_dict[i], 4)
                    celf_ep = (k, i, mg_r, 0)
                    celf_seq.append(celf_ep)
                    for celf_item in celf_seq:
                        if celf_ep[2] >= celf_item[2]:
                            celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                            celf_seq.pop()
                            break

        return celf_seq


class SeedSelectionNGAPPW:
    def __init__(self, g_dict, s_c_dict, prod_list, pw_list):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### pw_list: (list) the product weight list
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.pw_list = pw_list

    def generateCelfSequence(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) (k_prod, i_node, mg, flag)
        celf_seq = [(-1, '-1', 0.0, 0)]
        diffap_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeDict({i}, i, 1)
            ei = diffap_ss.getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    mg = round(ei * self.product_list[k][0] * self.pw_list[k], 4)
                    celf_ep = (k, i, mg, 0)
                    celf_seq.append(celf_ep)
                    for celf_item in celf_seq:
                        if celf_ep[2] >= celf_item[2]:
                            celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                            celf_seq.pop()
                            break

        return celf_seq

    def generateCelfSequenceR(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) (k_prod, i_node, mg, flag)
        celf_seq = [(-1, '-1', 0.0, 0)]
        diffap_ss = DiffusionAccProb(self.graph_dict, self.seed_cost_dict, self.product_list)

        for i in self.graph_dict:
            i_dict = diffap_ss.buildNodeDict({i}, i, 1)
            ei = diffap_ss.getExpectedInf(i_dict)

            if ei > 0:
                for k in range(self.num_product):
                    if self.seed_cost_dict[i] == 0:
                        break
                    else:
                        mg_r = round(ei * self.product_list[k][0] * self.pw_list[k] / self.seed_cost_dict[i], 4)
                    celf_ep = (k, i, mg_r, 0)
                    celf_seq.append(celf_ep)
                    for celf_item in celf_seq:
                        if celf_ep[2] >= celf_item[2]:
                            celf_seq.insert(celf_seq.index(celf_item), celf_ep)
                            celf_seq.pop()
                            break

        return celf_seq


if __name__ == '__main__':
    data_set_name = 'email_undirected'
    product_name = 'r1p3n1'
    cascade_model = 'ic'
    total_budget = 10
    distribution_type = 1
    whether_passing_information_without_purchasing = bool(0)
    pp_strategy = 1
    eva_monte_carlo = 1000

    iniG = IniGraph(data_set_name)
    iniP = IniProduct(product_name)

    seed_cost_dict = iniG.constructSeedCostDict()
    graph_dict = iniG.constructGraphDict(cascade_model)
    product_list = iniP.getProductList()
    num_node = len(seed_cost_dict)
    num_product = len(product_list)
    product_weight_list = getProductWeight(product_list, distribution_type)

    # -- initialization for each budget --
    start_time = time.time()
    ssngap = SeedSelectionNGAP(graph_dict, seed_cost_dict, product_list)
    diffap = DiffusionAccProb(graph_dict, seed_cost_dict, product_list)

    # -- initialization for each sample --
    now_budget, now_profit = 0.0, 0.0
    seed_set = [set() for _ in range(num_product)]
    expected_profit_k = [0.0 for _ in range(num_product)]

    celf_sequence = ssngap.generateCelfSequence()
    mep_g = celf_sequence.pop(0)
    mep_k_prod, mep_i_node, mep_mg, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]
    print(round(time.time() - start_time, 4))

    while now_budget < total_budget and mep_i_node != '-1':
        if now_budget + seed_cost_dict[mep_i_node] > total_budget:
            mep_g = celf_sequence.pop(0)
            mep_k_prod, mep_i_node, mep_mg, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]
            if mep_i_node == '-1':
                break
            continue

        seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
        if mep_flag == seed_set_length:
            print(len(celf_sequence), mep_g)
            now_profit = round(now_profit + mep_mg, 4)
            now_budget = round(now_budget + seed_cost_dict[mep_i_node], 2)
            seed_set[mep_k_prod].add(mep_i_node)
            expected_profit_k[mep_k_prod] = round(expected_profit_k[mep_k_prod] + mep_mg, 4)
            print(round(time.time() - start_time, 4), now_budget, now_profit, seed_set)
        else:
            seed_set_k = copy.deepcopy(seed_set[mep_k_prod])
            seed_set_k.add(mep_i_node)
            s_dict = {}
            for s in seed_set_k:
                mep_i_dict = diffap.buildNodeDict(seed_set_k, s, 1)
                for ii in mep_i_dict:
                    if ii not in s_dict:
                        s_dict[ii] = mep_i_dict[ii]
                    else:
                        s_dict[ii] += mep_i_dict[ii]
            expected_inf = diffap.getExpectedInf(s_dict)
            ep_g = round(expected_inf * product_list[mep_k_prod][0], 4)
            mg_g = round(ep_g - expected_profit_k[mep_k_prod], 4)
            ep_flag = seed_set_length

            if mg_g > 0:
                celf_ep_g = (mep_k_prod, mep_i_node, mg_g, ep_flag)
                celf_sequence.append(celf_ep_g)
                for celf_item_g in celf_sequence:
                    if celf_ep_g[2] >= celf_item_g[2]:
                        celf_sequence.insert(celf_sequence.index(celf_item_g), celf_ep_g)
                        celf_sequence.pop()
                        break

        mep_g = celf_sequence.pop(0)
        mep_k_prod, mep_i_node, mep_mg, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]

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