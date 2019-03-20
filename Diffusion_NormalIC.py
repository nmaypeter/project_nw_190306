from Initialization import *
import copy


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
        self.monte = monte

    def getExpectedProfit(self, k_prod, i_node, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        s_set_t[k_prod].add(i_node)
        a_n_set = copy.deepcopy(s_set_t)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        # -- notice: prevent the node from owing no receiver --
        if i_node not in self.graph_dict:
            return round(ep, 4)

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
                if i_node in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node]:
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

            if acc_prob > 0.0001:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, acc_prob * float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)

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

            if acc_prob > 0.0001:
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
    def __init__(self, g_dict, s_c_dict, prod_list, monte, p_w_list):
        ### g_dict: (dict) the graph
        ### s_c_dict: (dict) the set of cost for seeds
        ### prod_list: (list) the set to record products [kk's profit, kk's cost, kk's price]
        ### num_node: (int) the number of nodes
        ### num_product: (int) the kinds of products
        ### monte: (int) monte carlo times
        ### p_w_list: (list) the product weight list
        self.graph_dict = g_dict
        self.seed_cost_dict = s_c_dict
        self.product_list = prod_list
        self.num_node = len(s_c_dict)
        self.num_product = len(prod_list)
        self.monte = monte
        self.pw_list = p_w_list

    def getExpectedProfit(self, k_prod, i_node, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        s_set_t[k_prod].add(i_node)
        a_n_set = copy.deepcopy(s_set_t)
        a_e_set = [{} for _ in range(self.num_product)]
        ep = 0.0

        # -- notice: prevent the node from owing no receiver --
        if i_node not in self.graph_dict:
            return round(ep, 4)

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
                if i_node in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node]:
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

            if acc_prob > 0.0001:
                continue

            out_dict = self.graph_dict[i_node_t]
            for out in out_dict:
                if random.random() > float(out_dict[out]):
                    continue

                if out in a_n_set[k_prod_t]:
                    continue
                if i_node in a_e_set[k_prod_t] and out in a_e_set[k_prod_t][i_node]:
                    continue
                try_a_n_sequence.append([k_prod_t, out, acc_prob * float(out_dict[out])])
                a_n_set[k_prod_t].add(i_node_t)
                if i_node_t in a_e_set[k_prod_t]:
                    a_e_set[k_prod_t][i_node_t].add(out)
                else:
                    a_e_set[k_prod_t][i_node_t] = {out}

        return round(ep, 4)

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

            if acc_prob > 0.0001:
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

    def getExpectedProfit(self, k_prod, i_node, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        s_set_t[k_prod].add(i_node)
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
                        ii_prob = str(round(float(self.graph_dict[i][ii]), 4))

                        if ii not in i_dict:
                            i_dict[ii] = [ii_prob]
                        elif ii in i_dict:
                            i_dict[ii].append(ii_prob)

                    for ii in self.graph_dict[i]:
                        if ii in union_seed_set:
                            continue
                        if ii not in self.graph_dict:
                            continue
                        ii_prob = self.graph_dict[i][ii]

                        for iii in self.graph_dict[ii]:
                            if iii in union_seed_set:
                                continue
                            iii_prob = self.graph_dict[ii][iii]
                            iii_prob = str(round(float(ii_prob) * float(iii_prob), 4))
                            if iii not in i_dict:
                                i_dict[iii] = [iii_prob]
                            elif iii in i_dict:
                                i_dict[iii].append(iii_prob)

                    for ii in self.graph_dict[i]:
                        if ii in union_seed_set:
                            continue
                        if ii not in self.graph_dict:
                            continue
                        ii_prob = self.graph_dict[i][ii]

                        for iii in self.graph_dict[ii]:
                            if iii in union_seed_set:
                                continue
                            if iii not in self.graph_dict:
                                continue
                            iii_prob = self.graph_dict[ii][iii]
                            iii_prob = str(float(ii_prob) * float(iii_prob))

                            for iiii in self.graph_dict[iii]:
                                if iiii in union_seed_set:
                                    continue
                                iiii_prob = self.graph_dict[iii][iiii]
                                iiii_prob = str(round(float(iii_prob) * float(iiii_prob), 4))

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

                    for ii in self.graph_dict[i]:
                        if ii not in self.graph_dict:
                            continue
                        ii_prob = self.graph_dict[i][ii]

                        for iii in self.graph_dict[ii]:
                            if iii in union_seed_set:
                                continue
                            iii_prob = str(float(ii_prob) * float(self.graph_dict[ii][iii]))

                            if iii not in i_dict:
                                i_dict[iii] = [iii_prob]
                            elif iii in i_dict:
                                i_dict[iii].append(iii_prob)

                    for ii in self.graph_dict[i]:
                        if ii not in self.graph_dict:
                            continue
                        ii_prob = self.graph_dict[i][ii]

                        for iii in self.graph_dict[ii]:
                            if iii not in self.graph_dict:
                                continue
                            iii_prob = str(float(ii_prob) * float(self.graph_dict[ii][iii]))

                            for iiii in self.graph_dict[iii]:
                                if iiii in union_seed_set:
                                    continue
                                iiii_prob = str(float(iii_prob) * float(self.graph_dict[iii][iiii]))

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

    def getExpectedProfit(self, k_prod, i_node, s_set, s_i_dict):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        temp_s_i_dict = [{} for _ in range(self.num_product)]
        if i_node not in s_set[k_prod]:
            temp_s_i_dict = copy.deepcopy(s_i_dict)
            temp_union_seed_set = set()
            for k in range(self.num_product):
                temp_union_seed_set = temp_union_seed_set | s_set[k]

            if i_node in s_i_dict[k_prod]:
                for i_prob in s_i_dict[k_prod][i_node]:
                    if i_node in temp_s_i_dict[k_prod] and i_prob in temp_s_i_dict[k_prod][i_node]:
                        temp_s_i_dict[k_prod][i_node].remove(i_prob)
                    if i_node in self.graph_dict:
                        for ii in self.graph_dict[i_node]:
                            if ii in temp_union_seed_set:
                                continue
                            ii_prob = self.graph_dict[i_node][ii]
                            ii_prob = str(round(float(i_prob) * float(ii_prob), 4))

                            if ii in temp_s_i_dict[k_prod] and ii_prob in temp_s_i_dict[k_prod][ii]:
                                temp_s_i_dict[k_prod][ii].remove(ii_prob)

                        for ii in self.graph_dict[i_node]:
                            if ii not in self.graph_dict:
                                continue
                            ii_prob = self.graph_dict[i_node][ii]
                            ii_prob = str(float(i_prob) * float(ii_prob))

                            for iii in self.graph_dict[ii]:
                                if iii in temp_union_seed_set:
                                    continue
                                iii_prob = self.graph_dict[ii][iii]
                                iii_prob = str(round(float(ii_prob) * float(iii_prob), 4))
                                if iii in temp_s_i_dict[k_prod] and iii_prob in temp_s_i_dict[k_prod][iii]:
                                    temp_s_i_dict[k_prod][iii].remove(iii_prob)

                        for ii in self.graph_dict[i_node]:
                            if ii not in self.graph_dict:
                                continue
                            ii_prob = self.graph_dict[i_node][ii]
                            ii_prob = str(float(i_prob) * float(ii_prob))

                            for iii in self.graph_dict[ii]:
                                if iii not in self.graph_dict:
                                    continue
                                iii_prob = self.graph_dict[ii][iii]
                                iii_prob = str(float(ii_prob) * float(iii_prob))

                                for iiii in self.graph_dict[iii]:
                                    if iiii in temp_union_seed_set:
                                        continue
                                    iiii_prob = self.graph_dict[iii][iiii]
                                    iiii_prob = str(round(float(iii_prob) * float(iiii_prob), 4))

                                    if iiii in temp_s_i_dict[k_prod] and iiii_prob in temp_s_i_dict[k_prod][iiii]:
                                        temp_s_i_dict[k_prod][iiii].remove(iiii_prob)

        s_set_t = copy.deepcopy(s_set)
        s_set_t[k_prod].add(i_node)
        union_seed_set = set()
        for k in range(self.num_product):
            union_seed_set = union_seed_set | s_set_t[k]
        ep = 0.0

        if i_node in self.graph_dict:
            for ii in self.graph_dict[i_node]:
                if ii in union_seed_set:
                    continue
                ii_prob = str(round(float(self.graph_dict[i_node][ii]), 4))

                if ii not in temp_s_i_dict[k_prod]:
                    temp_s_i_dict[k_prod][ii] = [ii_prob]
                elif ii in temp_s_i_dict[k_prod]:
                    temp_s_i_dict[k_prod][ii].append(ii_prob)

            for ii in self.graph_dict[i_node]:
                if ii not in self.graph_dict:
                    continue
                ii_prob = self.graph_dict[i_node][ii]

                for iii in self.graph_dict[ii]:
                    if iii in union_seed_set:
                        continue
                    iii_prob = self.graph_dict[ii][iii]
                    iii_prob = str(round(float(ii_prob) * float(iii_prob), 4))
                    if iii not in temp_s_i_dict[k_prod]:
                        temp_s_i_dict[k_prod][iii] = [iii_prob]
                    elif iii in temp_s_i_dict[k_prod]:
                        temp_s_i_dict[k_prod][iii].append(iii_prob)

            for ii in self.graph_dict[i_node]:
                if ii not in self.graph_dict:
                    continue
                ii_prob = self.graph_dict[i_node][ii]

                for iii in self.graph_dict[ii]:
                    if iii not in self.graph_dict:
                        continue
                    iii_prob = self.graph_dict[ii][iii]
                    iii_prob = str(float(ii_prob) * float(iii_prob))

                    for iiii in self.graph_dict[iii]:
                        if iiii in union_seed_set:
                            continue
                        iiii_prob = self.graph_dict[iii][iiii]
                        iiii_prob = str(round(float(iii_prob) * float(iiii_prob), 4))

                        if iiii not in temp_s_i_dict[k_prod]:
                            temp_s_i_dict[k_prod][iiii] = [iiii_prob]
                        elif iiii in temp_s_i_dict[k_prod]:
                            temp_s_i_dict[k_prod][iiii].append(iiii_prob)

        for k in range(self.num_product):
            for i in temp_s_i_dict[k]:
                acc_prob = 1.0
                for prob in temp_s_i_dict[k][i]:
                    acc_prob *= (1 - float(prob))
                ep += ((1 - acc_prob) * self.product_list[k][0])

        return round(ep, 4), temp_s_i_dict


class DiffusionAccProbPW:
    def __init__(self, g_dict, s_c_dict, prod_list, p_w_list):
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
        self.pw_list = p_w_list

    def getExpectedProfit(self, k_prod, i_node, s_set):
        # -- calculate the expected profit for single node when i_node's chosen as a seed for k-product --
        ### ep: (float2) the expected profit
        s_set_t = copy.deepcopy(s_set)
        s_set_t[k_prod].add(i_node)
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
                        if ii not in i_dict:
                            i_dict[ii] = ['0.1']
                        elif ii in i_dict:
                            i_dict[ii].append('0.1')

                    for ii in self.graph_dict[i]:
                        if ii not in self.graph_dict:
                            continue
                        for iii in self.graph_dict[ii]:
                            if iii in union_seed_set:
                                continue
                            if iii not in i_dict:
                                i_dict[iii] = ['0.01']
                            elif iii in i_dict:
                                i_dict[iii].append('0.01')

                    for ii in self.graph_dict[i]:
                        if ii not in self.graph_dict:
                            continue
                        for iii in self.graph_dict[ii]:
                            if iii not in self.graph_dict:
                                continue
                            for iiii in self.graph_dict[ii]:
                                if iiii in union_seed_set:
                                    continue
                                if iiii not in i_dict:
                                    i_dict[iiii] = ['0.001']
                                elif iiii in i_dict:
                                    i_dict[iiii].append('0.001')

            for i in i_dict:
                acc_prob = 1.0
                for prob in i_dict[i]:
                    acc_prob *= (1 - float(prob))
                ep += ((1 - acc_prob) * self.product_list[k][0] * self.pw_list[k])

        return round(ep, 4)

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
                        if ii not in i_dict:
                            i_dict[ii] = ['0.1']
                        elif ii in i_dict:
                            i_dict[ii].append('0.1')

                    for ii in self.graph_dict[i]:
                        if ii not in self.graph_dict:
                            continue
                        for iii in self.graph_dict[ii]:
                            if iii in union_seed_set:
                                continue
                            if iii not in i_dict:
                                i_dict[iii] = ['0.01']
                            elif iii in i_dict:
                                i_dict[iii].append('0.01')

                    for ii in self.graph_dict[i]:
                        if ii not in self.graph_dict:
                            continue
                        for iii in self.graph_dict[ii]:
                            if iii not in self.graph_dict:
                                continue
                            for iiii in self.graph_dict[ii]:
                                if iiii in union_seed_set:
                                    continue
                                if iiii not in i_dict:
                                    i_dict[iiii] = ['0.001']
                                elif iiii in i_dict:
                                    i_dict[iiii].append('0.001')

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