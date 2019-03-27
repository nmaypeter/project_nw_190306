from Diffusion_NormalIC import *


class SeedSelectionDG:
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

    def IterativePrune(self):
        A_set_n = [set() for _ in range(self.num_product)]
        B_set_n = [set(self.graph_dict.keys()) for _ in range(self.num_product)]

        diff_ss = Diffusion(self.graph_dict, self.seed_cost_dict, self.product_list, self.monte)

        while 1:
            A_set_o = copy.deepcopy(A_set_n)
            B_set_o = copy.deepcopy(B_set_n)
            A_set_n, B_set_n = [set() for _ in range(self.num_product)], [set() for _ in range(self.num_product)]

            ep_a, ep_b = 0.0, 0.0
            for _ in range(self.monte):
                ep_a += diff_ss.getSeedSetProfit(A_set_o)
                ep_b += diff_ss.getSeedSetProfit(B_set_o)
            ep_a, ep_b = round(ep_a / self.monte, 4), round(ep_b / self.monte, 4)

            for k in range(self.num_product):
                for i in B_set_o[k]:
                    ep = 0.0
                    for _ in range(self.monte):
                        ep += diff_ss.getExpectedRemovedProfit(k, i, B_set_o)
                    ep = round(ep / self.monte, 4)
                    if (ep_b - ep) - self.seed_cost_dict[i] > 0:
                        A_set_n[k].add(i)

                for i in A_set_o[k]:
                    ep = 0.0
                    for _ in range(self.monte):
                        ep += diff_ss.getExpectedProfit(k, i, A_set_o)
                    ep = round(ep / self.monte, 4)
                    if (ep - ep_a) - self.seed_cost_dict[i] >= 0:
                        B_set_n[k].add(i)

            if A_set_o == A_set_n and B_set_o == B_set_n:
                return A_set_n, B_set_n

    def generateDGSet(self):
        # -- calculate expected profit for all combinations of nodes and products --
        ### celf_ep: (list) [k_prod, i_node, mg, flag]
        dg_s_set = [set() for _ in range(self.num_product)]
        dg_t_set = [set(self.graph_dict.keys()) for _ in range(self.num_product)]
        ep_s, ep_t = 0.0, 0.0

        diff_ss = Diffusion(self.graph_dict, self.seed_cost_dict, self.product_list, self.monte)

        for k in range(self.num_product):
            for i in set(self.graph_dict.keys()):
                s_set = copy.deepcopy(dg_s_set)
                ep = 0.0
                for _ in range(self.monte):
                    ep += diff_ss.getExpectedProfit(k, i, s_set)
                ep = round(ep / self.monte, 4)
                r_plus = round((ep - ep_s) - self.seed_cost_dict[i], 4)
                print(k, i, ep, ep_s, r_plus)

                t_set = copy.deepcopy(dg_t_set)
                ep = 0.0
                for _ in range(self.monte):
                    ep += diff_ss.getExpectedRemovedProfit(k, i, t_set)
                ep = round(ep / self.monte, 4)
                r_minus = round((ep_t - ep) - self.seed_cost_dict[i], 4)
                print(k, i, ep, ep_t, r_minus)

                if r_plus >= r_minus:
                    dg_s_set[k].add(i)
                    ep_s = 0.0
                    for _ in range(self.monte):
                        ep_s += diff_ss.getSeedSetProfit(dg_s_set)
                    ep_s = round(ep_s / self.monte, 4)
                else:
                    dg_t_set[k].remove(i)
                    ep_t = 0.0
                    for _ in range(self.monte):
                        ep_t += diff_ss.getSeedSetProfit(dg_t_set)
                    ep_t = round(ep_t / self.monte, 4)

        return dg_s_set, dg_t_set


if __name__ == '__main__':
    data_set_name = 'toy2'
    product_name = 'r1p3n1'
    distribution_type = 1
    whether_passing_information_without_purchasing = bool(0)
    pp_strategy = 1
    monte_carlo, eva_monte_carlo = 10, 1000

    iniG = IniGraph(data_set_name)
    iniP = IniProduct(product_name)

    seed_cost_dict = iniG.constructSeedCostDict()
    graph_dict = iniG.constructGraphDict()
    product_list = iniP.getProductList()
    num_node = len(seed_cost_dict)
    num_product = len(product_list)

    # -- initialization for each budget --
    start_time = time.time()
    ssdg = SeedSelectionDG(graph_dict, seed_cost_dict, product_list, monte_carlo)
    diff = Diffusion(graph_dict, seed_cost_dict, product_list, monte_carlo)

    # S, T = ssdg.generateDGSet()
    # print(S)
    # print(T)

    A, B = ssdg.IterativePrune()
    print(A)
    print(B)