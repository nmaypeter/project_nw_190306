from Initialization import *
import copy
import gc


def getProductWeight(prod_list, dis):
    price_list = [k[2] for k in prod_list]
    mu, sigma = 0, 1
    if dis == 1:
        mu = np.mean(price_list)
        sigma = (max(price_list) - mu) / 0.8415
    elif dis == 2:
        mu = sum(price_list)
        sigma = abs(min(price_list) - mu) / 3
    X = np.arange(0, 2, 0.001)
    Y = stats.norm.sf(X, mu, sigma)
    pw_list = [round(float(Y[np.argwhere(X == p)]), 4) for p in price_list]

    return pw_list


class Diffusion:
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
        self.prob_threshold = 0.001
        self.monte = monte

    def getSeedSetProfit(self, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        a_n_set = copy.deepcopy(s_set_t)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        # -- insert the children of seeds into try_s_n_sequence --
        ### try_s_n_sequence: (list) the sequence to store the seed for k-products [k, i]
        ### try_a_n_sequence: (list) the sequence to store the nodes may be activated for k-products [k, i, prob]
        try_s_n_sequence, try_a_n_sequence = [], []
        for k in range(self.num_product):
            for i in s_set_t[k]:
                try_s_n_sequence.append([k, i])

        while len(try_s_n_sequence) > 0:
            seed = choice(try_s_n_sequence)
            try_s_n_sequence.remove(seed)
            k_prod_t, i_node_t = seed[0], seed[1]

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        while len(try_a_n_sequence) > 0:
            try_node = choice(try_a_n_sequence)
            try_a_n_sequence.remove(try_node)
            k_prod_t, i_node_t, acc_prob = try_node[0], try_node[1], try_node[2]

            ### -- purchasing --
            ep += self.product_list[k_prod_t][0]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            if acc_prob < self.prob_threshold:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, acc_prob * float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)


class DiffusionPW:
    def __init__(self, g_dict, s_c_dict, prod_list, pw_list, monte):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### p_w_list: (list) the product weight list
        ### monte: (int) monte carlo times
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.pw_list = pw_list
        self.prob_threshold = 0.0001
        self.monte = monte

    def getSeedSetProfit(self, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        a_n_set = copy.deepcopy(s_set_t)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        # -- insert the children of seeds into try_s_n_sequence --
        ### try_s_n_sequence: (list) the sequence to store the seed for k-products [k, i]
        ### try_a_n_sequence: (list) the sequence to store the nodes may be activated for k-products [k, i, prob]
        try_s_n_sequence, try_a_n_sequence = [], []
        for k in range(self.num_product):
            for i in s_set_t[k]:
                try_s_n_sequence.append([k, i])

        while len(try_s_n_sequence) > 0:
            seed = choice(try_s_n_sequence)
            try_s_n_sequence.remove(seed)
            k_prod_t, i_node_t = seed[0], seed[1]

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        while len(try_a_n_sequence) > 0:
            try_node = choice(try_a_n_sequence)
            try_a_n_sequence.remove(try_node)
            k_prod_t, i_node_t, acc_prob = try_node[0], try_node[1], try_node[2]

            ### -- purchasing --
            ep += self.product_list[k_prod_t][0] * self.pw_list[k_prod_t]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            if acc_prob < self.prob_threshold:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, acc_prob * float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)


class DiffusionAccProb:
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

    def getSeedSetProfit(self, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        ep = 0.0

        for k in range(self.num_product):
            i_dict = {}
            for i in s_set_t[k]:
                if i in self.graph_dict:
                    for ii in self.graph_dict[i]:
                        if ii in s_set_t[k]:
                            continue
                        ii_prob = self.graph_dict[i][ii]

                        if ii not in i_dict:
                            i_dict[ii] = [ii_prob]
                        elif ii in i_dict:
                            i_dict[ii].append(ii_prob)

                        if ii not in self.graph_dict:
                            continue

                        for iii in self.graph_dict[ii]:
                            if iii in s_set_t[k]:
                                continue
                            iii_prob = str(round(float(ii_prob) * float(self.graph_dict[ii][iii]), 4))

                            if iii not in i_dict:
                                i_dict[iii] = [iii_prob]
                            elif iii in i_dict:
                                i_dict[iii].append(iii_prob)

                            if iii not in self.graph_dict:
                                continue

                            for iiii in self.graph_dict[iii]:
                                if iiii in s_set_t[k]:
                                    continue
                                iiii_prob = str(round(float(iii_prob) * float(self.graph_dict[iii][iiii]), 4))

                                if iiii not in i_dict:
                                    i_dict[iiii] = [iiii_prob]
                                elif iiii in i_dict:
                                    i_dict[iiii].append(iiii_prob)

            for i in i_dict:
                acc_prob = 1.0
                for prob in i_dict[i]:
                    acc_prob *= (1 - float(prob))
                ep += ((1 - acc_prob) * self.product_list[k][0])

        return round(ep, 4)


class DiffusionAccProb2:
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

    def generateIDict(self, i_node):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set = [set() for _ in range(self.num_product)]
        s_set[0].add(i_node)

        i_dict = {}
        if i_node in self.graph_dict:
            for ii in self.graph_dict[i_node]:
                if ii == i_node:
                    continue
                ii_prob = self.graph_dict[i_node][ii]

                if ii not in i_dict:
                    i_dict[ii] = [ii_prob]
                elif ii in i_dict:
                    i_dict[ii].append(ii_prob)

                if ii not in self.graph_dict:
                    continue

                for iii in self.graph_dict[ii]:
                    if iii == i_node:
                        continue
                    iii_prob = str(round(float(ii_prob) * float(self.graph_dict[ii][iii]), 4))

                    if iii not in i_dict:
                        i_dict[iii] = [iii_prob]
                    elif iii in i_dict:
                        i_dict[iii].append(iii_prob)

                    if iii not in self.graph_dict:
                        continue

                    for iiii in self.graph_dict[iii]:
                        if iiii == i_node:
                            continue
                        iiii_prob = str(round(float(iii_prob) * float(self.graph_dict[iii][iiii]), 4))

                        if iiii not in i_dict:
                            i_dict[iiii] = [iiii_prob]
                        elif iiii in i_dict:
                            i_dict[iiii].append(iiii_prob)

        ep = 0.0
        for i in i_dict:
            acc_prob = 1.0
            for prob in i_dict[i]:
                acc_prob *= (1 - float(prob))
            ep += ((1 - acc_prob) * self.product_list[0][0])

        return round(ep, 4), i_dict

    def getExpectedProfit(self, k_prod, i_node, s_set, s_i_dict, k_i_dict):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        temp_s_i_dict = [{} for _ in range(self.num_product)]
        temp_k_i_dict = copy.deepcopy(k_i_dict)
        temp_union_seed_set = set()
        for k in range(self.num_product):
            temp_union_seed_set = temp_union_seed_set | s_set[k]

        if i_node not in s_set[k_prod]:
            temp_s_i_dict = copy.deepcopy(s_i_dict)

            if i_node in s_i_dict[k_prod]:
                for i_prob in s_i_dict[k_prod][i_node]:
                    if i_node in temp_s_i_dict[k_prod] and i_prob in temp_s_i_dict[k_prod][i_node]:
                        temp_s_i_dict[k_prod][i_node].remove(i_prob)

                    if i_node not in self.graph_dict:
                        continue

                    for ii in self.graph_dict[i_node]:
                        if ii in temp_union_seed_set:
                            continue
                        ii_prob = self.graph_dict[i_node][ii]
                        ii_prob = str(round(float(i_prob) * float(ii_prob), 4))
                        if ii in temp_s_i_dict[k_prod] and ii_prob in temp_s_i_dict[k_prod][ii]:
                            temp_s_i_dict[k_prod][ii].remove(ii_prob)

                        if ii not in self.graph_dict:
                            continue

                        for iii in self.graph_dict[ii]:
                            if iii in temp_union_seed_set:
                                continue
                            iii_prob = self.graph_dict[ii][iii]
                            iii_prob = str(round(float(ii_prob) * float(iii_prob), 4))
                            if iii in temp_s_i_dict[k_prod] and iii_prob in temp_s_i_dict[k_prod][iii]:
                                temp_s_i_dict[k_prod][iii].remove(iii_prob)

                            if iii not in self.graph_dict:
                                continue

                            for iiii in self.graph_dict[iii]:
                                if iiii in temp_union_seed_set:
                                    continue
                                iiii_prob = self.graph_dict[iii][iiii]
                                iiii_prob = str(round(float(iii_prob) * float(iiii_prob), 4))

                                if iiii in temp_s_i_dict[k_prod] and iiii_prob in temp_s_i_dict[k_prod][iiii]:
                                    temp_s_i_dict[k_prod][iiii].remove(iiii_prob)

            for s_node in s_set[k_prod]:
                if s_node in k_i_dict:
                    for s_prob in k_i_dict[s_node]:
                        if s_node in temp_k_i_dict and s_prob in temp_k_i_dict[s_node]:
                            temp_k_i_dict[s_node].remove(s_prob)

                            if s_node not in self.graph_dict:
                                continue

                            for ss in self.graph_dict[s_node]:
                                ss_prob = self.graph_dict[s_node][ss]
                                ss_prob = str(round(float(s_prob) * float(ss_prob), 4))
                                if ss in temp_k_i_dict and ss_prob in temp_k_i_dict[ss]:
                                    temp_k_i_dict[ss].remove(ss_prob)

                                if ss not in self.graph_dict:
                                    continue

                                for sss in self.graph_dict[ss]:
                                    sss_prob = self.graph_dict[ss][sss]
                                    sss_prob = str(round(float(ss_prob) * float(sss_prob), 4))
                                    if sss in temp_k_i_dict and sss_prob in temp_k_i_dict[sss]:
                                        temp_k_i_dict[sss].remove(sss_prob)

                                    if sss not in self.graph_dict:
                                        continue

                                    for ssss in self.graph_dict[sss]:
                                        ssss_prob = self.graph_dict[sss][ssss]
                                        ssss_prob = str(round(float(sss_prob) * float(ssss_prob), 4))

                                        if ssss in temp_k_i_dict and ssss_prob in temp_k_i_dict[ssss]:
                                            temp_k_i_dict[ssss].remove(ssss_prob)

        for item in temp_k_i_dict:
            if item in temp_s_i_dict[k_prod]:
                temp_s_i_dict[k_prod][item] += temp_k_i_dict[item]
            else:
                temp_s_i_dict[k_prod][item] = temp_k_i_dict[item]

        ep = 0.0
        for k in range(self.num_product):
            for i in temp_s_i_dict[k]:
                acc_prob = 1.0
                for prob in temp_s_i_dict[k][i]:
                    acc_prob *= (1 - float(prob))
                ep += ((1 - acc_prob) * self.product_list[k][0])

        return round(ep, 4), temp_s_i_dict


class DiffusionAccProb3:
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
        self.prob_threshold = 0.001

    def getSeedSetNeighborProfit(self, s_set_k, i_node, i_acc_prob):
        i_dict = {}

        for i_non in self.graph_dict[i_node]:
            if i_non in s_set_k:
                continue
            i_non_prob = str(round(float(self.graph_dict[i_node][i_non]) * float(i_acc_prob), 4))

            if float(i_non_prob) < self.prob_threshold:
                continue

            if i_non not in i_dict:
                i_dict[i_non] = [i_non_prob]
            elif i_non in i_dict:
                i_dict[i_non].append(i_non_prob)

            if i_non in self.graph_dict:
                diff_d = DiffusionAccProb3(self.graph_dict, self.seed_cost_dict, self.product_list)
                i_non_dict = diff_d.getSeedSetNeighborProfit(s_set_k, i_non, i_non_prob)

                for item in i_non_dict:
                    if item in i_dict:
                        i_dict[item] += i_non_dict[item]
                    else:
                        i_dict[item] = i_non_dict[item]

        return i_dict

    def getSeedSetProfit(self, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        ep = 0.0

        diff_d = DiffusionAccProb3(self.graph_dict, self.seed_cost_dict, self.product_list)

        for k in range(self.num_product):
            s_dict = {}
            for i in s_set_t[k]:
                if i in self.graph_dict:
                    i_dict = diff_d.getSeedSetNeighborProfit(s_set_t[k], i, '1')

                    for item in i_dict:
                        if item in s_dict:
                            s_dict[item] += i_dict[item]
                        else:
                            s_dict[item] = i_dict[item]
            for i in s_dict:
                acc_prob = 1.0
                for prob in s_dict[i]:
                    acc_prob *= (1 - float(prob))
                ep += ((1 - acc_prob) * self.product_list[k][0])

        return round(ep, 4)


class DiffusionAccProb4:
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
        self.prob_threshold = 0.001

    def getSeedSetNeighborProfit(self, union_seed_set, i_node, i_acc_prob, b_dict):
        i_dict = {}

        if i_node in b_dict and not list(set(union_seed_set).intersection(set(b_dict[i_node]))):
            for item in b_dict[i_node]:
                if item in union_seed_set:
                    continue
                for item_prob in b_dict[i_node][item]:
                    item_acc_prob = str(round(float(item_prob) * float(i_acc_prob), 4))

                    if float(item_acc_prob) < self.prob_threshold:
                        continue
                    if item in i_dict:
                        i_dict[item].append(item_acc_prob)
                    else:
                        i_dict[item] = [item_acc_prob]
        else:
            for i_non in self.graph_dict[i_node]:
                if i_non in union_seed_set:
                    continue
                i_non_prob = str(round(float(self.graph_dict[i_node][i_non]) * float(i_acc_prob), 4))

                if float(i_non_prob) < self.prob_threshold:
                    continue

                if i_non not in i_dict:
                    i_dict[i_non] = [i_non_prob]
                elif i_non in i_dict:
                    i_dict[i_non].append(i_non_prob)

                if i_non in b_dict and not list(set(union_seed_set).intersection(set(b_dict[i_non]))):
                    for item in b_dict[i_non]:
                        if item in union_seed_set:
                            continue
                        for item_prob in b_dict[i_non][item]:
                            item_acc_prob = str(round(float(item_prob) * float(i_non_prob), 4))

                            if float(item_acc_prob) < 0.001:
                                continue
                            if item in i_dict:
                                i_dict[item].append(item_acc_prob)
                            else:
                                i_dict[item] = [item_acc_prob]
                else:
                    if i_non in self.graph_dict:
                        diff_d = DiffusionAccProb4(self.graph_dict, self.seed_cost_dict, self.product_list)
                        i_non_dict = diff_d.getSeedSetNeighborProfit(union_seed_set, i_non, i_non_prob, b_dict)

                        for item in i_non_dict:
                            if item in i_dict:
                                i_dict[item] += i_non_dict[item]
                            else:
                                i_dict[item] = i_non_dict[item]

        return i_dict

    def getSeedSetProfit(self, s_set, b_dict):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        union_seed_set = set()
        for k in range(self.num_product):
            union_seed_set = union_seed_set | s_set_t[k]
        ep = 0.0

        diff_d = DiffusionAccProb4(self.graph_dict, self.seed_cost_dict, self.product_list)

        for k in range(self.num_product):
            s_dict = {}
            for i in s_set_t[k]:
                if i in self.graph_dict:
                    i_dict = diff_d.getSeedSetNeighborProfit(union_seed_set, i, '1', b_dict)

                    if i not in b_dict:
                        b_dict[i] = i_dict

                    for item in i_dict:
                        if item in s_dict:
                            s_dict[item] += i_dict[item]
                        else:
                            s_dict[item] = i_dict[item]
            for i in s_dict:
                acc_prob = 1.0
                for prob in s_dict[i]:
                    acc_prob *= (1 - float(prob))
                ep += ((1 - acc_prob) * self.product_list[k][0])

        return round(ep, 4), b_dict


class DiffusionAccProb5:
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
        self.prob_threshold = 0.001

    def buildNodeTree(self, seed, i_node, i_acc_prob):
        i_tree, i_descendants = {}, []

        if i_node in self.graph_dict:
            diff_d = DiffusionAccProb5(self.graph_dict, self.seed_cost_dict, self.product_list)
            for i_non in self.graph_dict[i_node]:
                if i_non == seed:
                    continue
                i_non_prob = str(round(float(self.graph_dict[i_node][i_non]) * float(i_acc_prob), 4))
                i_tree[i_non] = [i_non_prob, {}]
                if i_non not in i_descendants:
                    i_descendants.append(i_non)

                if float(i_non_prob) > self.prob_threshold:
                    i_non_tree, i_non_descendants = diff_d.buildNodeTree(seed, i_non, i_non_prob)
                    i_tree[i_non][1] = i_non_tree

                    for item in i_non_descendants:
                        if item not in i_descendants:
                            i_descendants.append(item)

        return i_tree, i_descendants

    def removeKIFromSITree(self, i_node, s_i_tree):
        temp_s_i_tree = {}
        diff_d = DiffusionAccProb5(self.graph_dict, self.seed_cost_dict, self.product_list)
        for i in s_i_tree:
            if i == i_node:
                continue
            else:
                if len(s_i_tree[i][1]) == 0:
                    temp_s_i_tree[i] = [s_i_tree[i][0], {}]
                    continue
                temp_s_i_tree[i] = [s_i_tree[i][0], diff_d.removeKIFromSITree(i_node, s_i_tree[i][1])]

        return temp_s_i_tree

    def removeSIFromKITree(self, s_set_k, k_i_tree):
        temp_k_i_tree = {}
        diff_d = DiffusionAccProb5(self.graph_dict, self.seed_cost_dict, self.product_list)
        for i in k_i_tree:
            if len(s_set_k.intersection(set(list(k_i_tree.keys())))) != 0:
                for s in s_set_k:
                    if s == i:
                        continue
            else:
                if len(k_i_tree[i][1]) == 0:
                    temp_k_i_tree[i] = [k_i_tree[i][0], {}]
                    continue
                temp_k_i_tree[i] = [k_i_tree[i][0], diff_d.removeKIFromSITree(s_set_k, k_i_tree[i][1])]

        return temp_k_i_tree

    def generateIDictUsingITree(self, i_tree):
        i_dict = [{} for _ in range(self.num_product)]
        for k in range(self.num_product):
            for s in i_tree[k]:
                for i in i_tree[k][s]:
                    i_dict_seq = [[i, i_tree[k][s][i][0], i_tree[k][s][i][1]]]

                    while len(i_dict_seq) != 0:
                        # print(len(i_dict_seq))
                        i_node, i_prob, i_subtree = i_dict_seq.pop()
                        if i_node in i_dict[k]:
                            i_dict[k][i_node].append(i_prob)
                        else:
                            i_dict[k][i_node] = [i_prob]

                        for i in i_subtree:
                            i_dict_seq.append([i, i_subtree[i][0], i_subtree[i][1]])

        return i_dict, i_tree

    @staticmethod
    def getExpectedInfluence(i_tree):
        i_dict = {}
        i_dict_seq = []
        for i in i_tree:
            i_dict_seq.append([i, i_tree[i][0], i_tree[i][1]])

        while len(i_dict_seq) != 0:
            i_node, i_prob, i_subtree = i_dict_seq.pop(0)
            if i_node in i_dict:
                i_dict[i_node].append(i_prob)
            else:
                i_dict[i_node] = [i_prob]

            for i in i_subtree:
                i_dict_seq.append([i, i_subtree[i][0], i_subtree[i][1]])

        ei = 0.0
        for i in i_dict:
            acc_prob = 1.0
            for prob in i_dict[i]:
                acc_prob *= (1 - float(prob))
            ei += (1 - acc_prob)

        return round(ei, 4)

    def getExpectedProfit(self, k_prod, i_node, s_set, s_i_tree, s_i_des, k_i_tree, k_i_des):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        temp_s_i_tree = s_i_tree
        temp_s_i_des = s_i_des
        temp_k_i_tree = [{} for _ in range(self.num_product)]
        temp_k_i_tree[k_prod] = k_i_tree
        temp_k_i_des = [[] for _ in range(self.num_product)]
        temp_k_i_des[k_prod] = k_i_des
        diff_d = DiffusionAccProb5(self.graph_dict, self.seed_cost_dict, self.product_list)

        if i_node in s_i_des[k_prod]:
            temp_s_i_tree, temp_s_i_des = [{} for _ in range(self.num_product)], [[] for _ in range(self.num_product)]
            for k in range(self.num_product):
                if k == k_prod:
                    for i in s_i_tree[k_prod]:
                        temp_s_i_tree[k][i] = diff_d.removeKIFromSITree(i_node, s_i_tree[k_prod][i])
                else:
                    temp_s_i_tree[k] = s_i_tree[k]
                    temp_s_i_des[k] = s_i_des[k]

        if len(s_set[k_prod].intersection(set(temp_k_i_des[k_prod]))) != 0:
            temp_k_i_tree, temp_k_i_des = [{} for _ in range(self.num_product)], [[] for _ in range(self.num_product)]
            for k in range(self.num_product):
                if k == k_prod:
                    for i in k_i_tree:
                        temp_k_i_tree[k_prod][i] = diff_d.removeSIFromKITree(s_set[k_prod], k_i_tree[i])
                else:
                    temp_k_i_tree[k] = k_i_tree
                    temp_k_i_des[k] = k_i_des

        if len(temp_s_i_tree[k_prod]) == 0:
            temp_s_i_tree[k_prod] = temp_k_i_tree[k_prod]
        else:
            temp_s_i_tree[k_prod][i_node] = temp_k_i_tree[k_prod][i_node]

        # i_dict = [{} for _ in range(self.num_product)]
        # i_tree = temp_s_i_tree
        # for k in range(self.num_product):
        #     for i in temp_s_i_tree[k]:
        #         k_i_dict = diff_d.generateIDictUsingITree()
        i_dict, i_tree = diff_d.generateIDictUsingITree(temp_s_i_tree)
        del temp_s_i_tree, temp_s_i_des, temp_k_i_tree, temp_k_i_des
        gc.collect()

        ep = 0.0
        for k in range(self.num_product):
            for i in i_dict[k]:
                acc_prob = 1.0
                for prob in i_dict[k][i]:
                    acc_prob *= (1 - float(prob))
                ep += ((1 - acc_prob) * self.product_list[k][0])

        i_des = [[] for _ in range(self.num_product)]
        for k in range(self.num_product):
            for i in i_dict[k]:
                i_des[k].append(i)

        return round(ep, 4), i_tree, i_des


class DiffusionAccProb6:
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
        self.prob_threshold = 0.001

    def buildNodeTree(self, seed, i_node, i_acc_prob):
        i_tree, i_dict = {}, {}

        if i_node in self.graph_dict:
            diff_d = DiffusionAccProb6(self.graph_dict, self.seed_cost_dict, self.product_list)
            for i_non in self.graph_dict[i_node]:
                if i_non == seed:
                    continue
                i_non_prob = str(round(float(self.graph_dict[i_node][i_non]) * float(i_acc_prob), 4))
                i_tree[i_non] = [i_non_prob, {}]
                if i_non not in i_dict:
                    i_dict[i_non] = [i_non_prob]
                else:
                    i_dict[i_non].append(i_non_prob)

                if float(i_non_prob) > self.prob_threshold:
                    if i_non not in self.graph_dict:
                        continue
                    i_non_tree, i_non_dict = diff_d.buildNodeTree(seed, i_non, i_non_prob)
                    i_tree[i_non][1] = i_non_tree
                    for item in i_non_dict:
                        if item not in i_dict:
                            i_dict[item] = i_non_dict[item]
                        else:
                            i_dict[item] += i_non_dict[item]

        return i_tree, i_dict

    @staticmethod
    def whetherNodeInTree(i_node, i_tree):
        i_tree_str = str(i_tree).replace('[\'1\',', '').replace('\'', '').replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace(',', '').replace(':', '')
        for i in i_tree_str.split():
            if i.isdigit():
                if i == i_node:
                    return True
        return False

    def removeNodeFromTree(self, i_node, s_i_tree):
        temp_s_i_tree = {}
        diff_d = DiffusionAccProb6(self.graph_dict, self.seed_cost_dict, self.product_list)
        for i in s_i_tree:
            if i == i_node:
                continue
            else:
                if len(s_i_tree[i][1]) == 0:
                    temp_s_i_tree[i] = [s_i_tree[i][0], {}]
                    continue

                if diff_d.whetherNodeInTree(i_node, s_i_tree[i][1]):
                    if i not in temp_s_i_tree:
                        temp_s_i_tree[i] = [s_i_tree[i][0], {}]
                    for ii in s_i_tree[i][1]:
                        if ii == i_node:
                            continue
                        else:
                            if len(s_i_tree[i][1][ii][1]) == 0:
                                temp_s_i_tree[i][1][ii] = [s_i_tree[i][1][ii][0], {}]
                                continue

                            if diff_d.whetherNodeInTree(i_node, s_i_tree[i][1][ii][1]):
                                if ii not in temp_s_i_tree[i]:
                                    temp_s_i_tree[i][1][ii] = [s_i_tree[i][1][ii][0], {}]
                                for iii in s_i_tree[i][1][ii][1]:
                                    if iii == i_node:
                                        continue
                                    else:
                                        if len(s_i_tree[i][1][ii][1][iii][1]) == 0:
                                            temp_s_i_tree[i][1][ii][1][iii] = [s_i_tree[i][1][ii][1][iii][0], {}]
                                            continue

                                        if i_node in diff_d.whetherNodeInTree(i_node, s_i_tree[i][1][ii][1][iii][1]):
                                            temp_s_i_tree[i][1][ii][1][iii] = [s_i_tree[i][1][ii][1][iii][0], diff_d.removeNodeFromTree(i_node, s_i_tree[i][1][ii][1][iii][1])]
                                        else:
                                            temp_s_i_tree[i][1][ii][1][iii] = s_i_tree[i][1][ii][1][iii]
                            else:
                                temp_s_i_tree[i][1][ii] = s_i_tree[i][1][ii]
                else:
                    temp_s_i_tree[i] = s_i_tree[i]

        return temp_s_i_tree

    def generateIDictUsingITree(self, i_tree):
        i_dict = [{} for _ in range(self.num_product)]
        for k in range(self.num_product):
            for s in i_tree[k]:
                for i in i_tree[k][s]:
                    i_dict_seq = [[i, i_tree[k][s][i][0], i_tree[k][s][i][1]]]

                    while len(i_dict_seq) != 0:
                        i_node, i_prob, i_subtree = i_dict_seq.pop()
                        if i_node in i_dict[k]:
                            i_dict[k][i_node].append(i_prob)
                        else:
                            i_dict[k][i_node] = [i_prob]

                        for i_n in i_subtree:
                            i_dict_seq.append([i_n, i_subtree[i_n][0], i_subtree[i_n][1]])

        return i_dict, i_tree

    def getExpectedProfit(self, k_prod, i_node, s_set, s_i_tree, k_i_tree):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        temp_s_i_tree = s_i_tree
        temp_k_i_tree = [{} for _ in range(self.num_product)]
        temp_k_i_tree[k_prod] = k_i_tree
        diff_d = DiffusionAccProb6(self.graph_dict, self.seed_cost_dict, self.product_list)

        if diff_d.whetherNodeInTree(i_node, s_i_tree[k_prod]):
            temp_s_i_tree = [{} for _ in range(self.num_product)]
            for k in range(self.num_product):
                if k == k_prod:
                    for i in s_i_tree[k_prod]:
                        temp_s_i_tree[k_prod][i] = diff_d.removeNodeFromTree(i_node, s_i_tree[k_prod][i])
                else:
                    temp_s_i_tree[k] = s_i_tree[k]

        for s in s_set[k_prod]:
            if diff_d.whetherNodeInTree(s, k_i_tree):
                temp_k_i_tree = [{} for _ in range(self.num_product)]
                for i in k_i_tree:
                    temp_k_i_tree[k_prod][i] = k_i_tree[i]
                    for s_node in s_set[k_prod]:
                        temp_k_i_tree[k_prod][i] = diff_d.removeNodeFromTree(s_node, temp_k_i_tree[k_prod][i])

        if len(temp_s_i_tree[k_prod]) == 0:
            temp_s_i_tree[k_prod] = temp_k_i_tree[k_prod]
        else:
            temp_s_i_tree[k_prod][i_node] = temp_k_i_tree[k_prod][i_node]

        i_dict, i_tree = diff_d.generateIDictUsingITree(temp_s_i_tree)

        ep = 0.0
        for k in range(self.num_product):
            for i in i_dict[k]:
                acc_prob = 1.0
                for prob in i_dict[k][i]:
                    acc_prob *= (1 - float(prob))
                ep += ((1 - acc_prob) * self.product_list[k][0])

        return round(ep, 4), i_tree


class DiffusionAccProbPW:
    def __init__(self, g_dict, s_c_dict, prod_list, pw_list):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### p_w_list: (list) the product weight list
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.pw_list = pw_list

    def getSeedSetProfit(self, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        union_seed_set = set()
        for k in range(self.num_product):
            union_seed_set = union_seed_set | s_set_t[k]
        ep = 0.0

        for k in range(self.num_product):
            i_dict = {}
            for i in s_set_t[k]:
                if i in self.graph_dict:
                    for ii in self.graph_dict[i]:
                        if ii in union_seed_set:
                            continue
                        ii_prob = self.graph_dict[i][ii]

                        if ii not in i_dict:
                            i_dict[ii] = [ii_prob]
                        elif ii in i_dict:
                            i_dict[ii].append(ii_prob)

                        if ii not in self.graph_dict:
                            continue

                        for iii in self.graph_dict[ii]:
                            if iii in union_seed_set:
                                continue
                            iii_prob = str(round(float(ii_prob) * float(self.graph_dict[ii][iii]), 4))

                            if iii not in i_dict:
                                i_dict[iii] = [iii_prob]
                            elif iii in i_dict:
                                i_dict[iii].append(iii_prob)

                            if iii not in self.graph_dict:
                                continue

                            for iiii in self.graph_dict[iii]:
                                if iiii in union_seed_set:
                                    continue
                                iiii_prob = str(round(float(iii_prob) * float(self.graph_dict[iii][iiii]), 4))

                                if iiii not in i_dict:
                                    i_dict[iiii] = [iiii_prob]
                                elif iiii in i_dict:
                                    i_dict[iiii].append(iiii_prob)

            for i in i_dict:
                acc_prob = 1.0
                for prob in i_dict[i]:
                    acc_prob *= (1 - float(prob))
                ep += ((1 - acc_prob) * self.product_list[k][0] * self.pw_list[k])

        return round(ep, 4)


class Evaluation:
    def __init__(self, g_dict, s_c_dict, prod_list, pps, wpiwp):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### pps: (int) the strategy to update personal prob.
        ### wpiwp: (bool) whether passing the information without purchasing
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.pps = pps
        self.wpiwp = wpiwp

    def setPersonalProbList(self, w_list):
        # -- according to pps, initialize the pp_list --
        # -- if the node i can't purchase the product k, then pp_list[k][i] = 0 --
        ### pp_list: (list) the list of personal prob. for all combinations of nodes and products
        ### pp_list[k]: (list) the list of personal prob. for k-product
        ### pp_list[k][i]: (float2) the personal prob. for i-node for k-product
        pp_list = [[1.0 for _ in range(self.num_node)] for _ in range(self.num_product)]

        for k in range(self.num_product):
            for i in range(self.num_node):
                if w_list[i] < self.product_list[k][2]:
                    pp_list[k][i] = 0

        for k in range(self.num_product):
            prod_price = self.product_list[k][2]
            for i in self.seed_cost_dict:
                if pp_list[k][int(i)] != 0:
                    if self.pps == 1:
                        # -- after buying a product, the prob. to buy another product will decrease randomly --
                        pp_list[k][int(i)] = round(random.uniform(0, pp_list[k][int(i)]), 4)
                    elif self.pps == 2:
                        # -- choose as expensive as possible --
                        pp_list[k][int(i)] *= round((prod_price / w_list[int(i)]), 4)
                    elif self.pps == 3:
                        # -- choose as cheap as possible --
                        pp_list[k][int(i)] *= round(1 - (prod_price / w_list[int(i)]), 4)

        return pp_list

    def updatePersonalProbList(self, k_prod, i_node, w_list, pp_list):
        prod_price = self.product_list[k_prod][2]
        if self.pps == 1:
            # -- after buying a product, the prob. to buy another product will decrease randomly --
            for k in range(self.num_product):
                if k == k_prod or w_list[int(i_node)] == 0.0:
                    pp_list[k][int(i_node)] = 0
                else:
                    pp_list[k][int(i_node)] = round(random.uniform(0, pp_list[k][int(i_node)]), 4)
        elif self.pps == 2:
            # -- choose as expensive as possible --
            for k in range(self.num_product):
                if k == k_prod or w_list[int(i_node)] == 0.0:
                    pp_list[k][int(i_node)] = 0
                else:
                    pp_list[k][int(i_node)] *= round((prod_price / w_list[int(i_node)]), 4)
        elif self.pps == 3:
            # -- choose as cheap as possible --
            for k in range(self.num_product):
                if k == k_prod or w_list[int(i_node)] == 0.0:
                    pp_list[k][int(i_node)] = 0
                else:
                    pp_list[k][int(i_node)] *= round(1 - (prod_price / w_list[int(i_node)]), 4)

        for k in range(self.num_product):
            for i in range(self.num_node):
                if w_list[i] < self.product_list[k][2]:
                    pp_list[k][i] = 0.0

        return pp_list

    def getSeedSetProfit(self, s_set, w_list, pp_list):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_total_set = set()
        for k in range(self.num_product):
            s_total_set = s_total_set.union(s_set[k])
        purc_set = [set() for _ in range(self.num_product)]
        a_n_set = copy.deepcopy(s_set)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        pro_k_list, pnn_k_list = [0.0 for _ in range(self.num_product)], [0 for _ in range(self.num_product)]

        # -- insert the children of seeds into try_s_n_sequence --
        ### try_s_n_sequence: (list) the sequence to store the seed for k-products [k, i]
        ### try_a_n_sequence: (list) the sequence to store the nodes may be activated for k-products [k, i, prob]
        try_s_n_sequence, try_a_n_sequence = [], []
        for k in range(self.num_product):
            for i in s_set[k]:
                try_s_n_sequence.append([k, i])

        while len(try_s_n_sequence) > 0:
            seed = choice(try_s_n_sequence)
            try_s_n_sequence.remove(seed)
            k_prod_t, i_node_t = seed[0], seed[1]

            # -- notice: prevent the node from owing no receiver --
            if i_node_t not in self.graph_dict:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in s_total_set:
                    continue
                if out in a_n_set[k_prod_t]:
                    continue
                if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                    continue
                if pp_list[k_prod_t][int(out)] == 0:
                    continue
                try_a_n_sequence.append([k_prod_t, out, self.graph_dict[i_node_t][out]])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

            # -- activate the nodes --
            eva = Evaluation(self.graph_dict, self.seed_cost_dict, self.product_list, self.pps, self.wpiwp)

            while len(try_a_n_sequence) > 0:
                try_node = choice(try_a_n_sequence)
                try_a_n_sequence.remove(try_node)
                k_prod_t, i_node_t, i_prob_t = try_node[0], try_node[1], try_node[2]
                dp = bool(0)

                ### -- whether purchasing or not --
                if random.random() <= pp_list[k_prod_t][int(i_node_t)]:
                    purc_set[k_prod_t].add(i_node_t)
                    a_n_set[k_prod_t].add(i_node_t)
                    w_list[int(i_node_t)] -= self.product_list[k_prod_t][2]
                    pp_list = eva.updatePersonalProbList(k_prod_t, i_node_t, w_list, pp_list)
                    ep += self.product_list[k_prod_t][0]
                    dp = bool(1)

                    pro_k_list[k_prod_t] += self.product_list[k_prod_t][0]

                if i_node_t not in self.graph_dict:
                    continue

                ### -- whether passing the information or not --
                if self.wpiwp or dp:
                    out_dict = self.graph_dict[i_node_t]
                    for out in out_dict:
                        if random.random() > float(out_dict[out]):
                            continue

                        if out in s_total_set:
                            continue
                        if out in a_n_set[k_prod_t]:
                            continue
                        if i_node_t in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node_t]:
                            continue
                        if pp_list[k_prod_t][int(out)] == 0:
                            continue
                        try_a_n_sequence.append([k_prod_t, out, self.graph_dict[i_node_t][out]])
                        a_n_set[k_prod_t].add(i_node_t)
                        if i_node_t in a_e_set[k_prod_t]:
                            a_e_set[k_prod_t][i_node_t].add(out)
                        else:
                            a_e_set[k_prod_t][i_node_t] = {out}

        for k in range(self.num_product):
            pro_k_list[k] = round(pro_k_list[k], 2)
            pnn_k_list[k] = round(len(purc_set[k]), 2)

        return round(ep, 2), pro_k_list, pnn_k_list