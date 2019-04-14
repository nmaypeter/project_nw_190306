from SeedSelection_NGAP import *

if __name__ == '__main__':
    wpiwp_seq = [bool(0), bool(1)]
    dis_seq = [1, 2]
    data_setting_seq = [1]
    cm_seq = [1, 2]
    prod_setting_seq, prod_setting2_seq = [1, 2], [1, 2, 3]
    sample_number = 1
    total_budget = 10
    pps_seq = [1, 2, 3]
    eva_monte_carlo = 100
    for wpiwp in wpiwp_seq:
        for distribution_type in dis_seq:
            for data_setting in data_setting_seq:
                data_set_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                                'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
                for cm in cm_seq:
                    cas_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
                    for prod_setting in prod_setting_seq:
                        for prod_setting2 in prod_setting2_seq:
                            product_name = 'r1p3n' + str(prod_setting) + 'a' * (prod_setting2 == 2) + 'b' * (prod_setting2 == 3)

                            iniG = IniGraph(data_set_name)
                            iniP = IniProduct(product_name)

                            seed_cost_dict = iniG.constructSeedCostDict()
                            graph_dict = iniG.constructGraphDict(cas_model)
                            product_list = iniP.getProductList()
                            num_node = len(seed_cost_dict)
                            num_product = len(product_list)
                            product_weight_list = getProductWeight(product_list, distribution_type)

                            seed_set_sequence, ss_time_sequence = [[] for _ in range(total_budget)], [[] for _ in range(total_budget)]
                            ssngappw_main = SeedSelectionNGAPPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
                            diffappw_main = DiffusionAccProbPW(graph_dict, seed_cost_dict, product_list, product_weight_list)
                            for sample_count in range(sample_number):
                                ss_strat_time = time.time()
                                begin_budget = 1
                                now_budget, now_profit = 0.0, 0.0
                                app_now_profit = 0.0
                                now_s_i_tree = [{} for _ in range(num_product)]
                                seed_set = [set() for _ in range(num_product)]
                                celf_sequence, i_tree_dict, app_now_s_i_tree = ssngappw_main.generateCelfSequenceR()
                                ss_acc_time = round(time.time() - ss_strat_time, 2)
                                temp_sequence = [[begin_budget, now_budget, now_profit, app_now_profit, now_s_i_tree, seed_set, celf_sequence, app_now_s_i_tree, ss_acc_time]]
                                while len(temp_sequence) != 0:
                                    ss_strat_time = time.time()
                                    begin_budget, now_budget, now_profit, app_now_profit, now_s_i_tree, seed_set, celf_sequence, app_now_s_i_tree, ss_acc_time = temp_sequence.pop(0)
                                    print('@ mngaprpwic seed selection @ data_set_name = ' + data_set_name + '_' + cas_model + ', dis = ' + str(distribution_type) + ', wpiwp = ' + str(wpiwp) +
                                          ', product_name = ' + product_name + ', budget = ' + str(begin_budget) + ', sample_count = ' + str(sample_count))
                                    mep_g = celf_sequence.pop(0)
                                    mep_k_prod, mep_i_node, mep_profit, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]

                                    while now_budget < begin_budget and mep_i_node != '-1':
                                        sc = seed_cost_dict[mep_i_node]
                                        if now_budget + sc >= begin_budget and begin_budget < total_budget and len(temp_sequence) == 0:
                                            ss_time = round(time.time() - ss_strat_time + ss_acc_time, 2)
                                            temp_celf_sequence = copy.deepcopy(celf_sequence)
                                            temp_celf_sequence.insert(0, mep_g)
                                            temp_sequence.append([begin_budget + 1, now_budget, now_profit, copy.deepcopy(app_now_profit), copy.deepcopy(now_s_i_tree), copy.deepcopy(seed_set),
                                                                  temp_celf_sequence, app_now_s_i_tree, ss_time])

                                        if now_budget + sc > begin_budget:
                                            mep_g = celf_sequence.pop(0)
                                            mep_k_prod, mep_i_node, mep_profit, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]
                                            if mep_i_node == '-1':
                                                break
                                            continue

                                        seed_set_length = sum(len(seed_set[kk]) for kk in range(num_product))
                                        if mep_flag == seed_set_length:
                                            now_profit = round(now_profit + mep_profit * seed_cost_dict[mep_i_node], 4)
                                            now_budget = round(now_budget + seed_cost_dict[mep_i_node], 2)
                                            now_s_i_tree = app_now_s_i_tree
                                            app_now_profit = 0.0
                                            seed_set[mep_k_prod].add(mep_i_node)
                                        else:
                                            ep_g, s_i_tree_g = diffappw_main.getExpectedProfit(mep_k_prod, mep_i_node, seed_set, now_s_i_tree, i_tree_dict[mep_i_node])
                                            mg_g = round(ep_g - now_profit, 4)
                                            if seed_cost_dict[mep_i_node] == 0:
                                                mg_ratio_g = 0
                                            else:
                                                mg_ratio_g = round(mg_g / seed_cost_dict[mep_i_node], 4)
                                            ep_flag = seed_set_length
                                            if mg_ratio_g >= app_now_profit:
                                                app_now_profit = mg_ratio_g
                                                app_now_s_i_tree = s_i_tree_g

                                            if mg_ratio_g > 0:
                                                celf_ep_g = (mep_k_prod, mep_i_node, mg_ratio_g, ep_flag)
                                                celf_sequence.append(celf_ep_g)
                                                for celf_item_g in celf_sequence:
                                                    if celf_ep_g[2] >= celf_item_g[2]:
                                                        celf_sequence.insert(celf_sequence.index(celf_item_g), celf_ep_g)
                                                        celf_sequence.pop()
                                                        break

                                        mep_g = celf_sequence.pop(0)
                                        mep_k_prod, mep_i_node, mep_profit, mep_flag = mep_g[0], mep_g[1], mep_g[2], mep_g[3]

                                    ss_time = round(time.time() - ss_strat_time + ss_acc_time, 2)
                                    print('ss_time = ' + str(ss_time) + 'sec')
                                    seed_set_sequence[begin_budget - 1].append(copy.deepcopy(seed_set))
                                    ss_time_sequence[begin_budget - 1].append(ss_time)

                            eva_start_time = time.time()
                            for bud in range(1, total_budget + 1):
                                result = [[] for _ in range(len(pps_seq))]
                                for pps in pps_seq:
                                    pps_start_time = time.time()
                                    avg_pro, avg_bud = 0.0, 0.0
                                    avg_sn_k, avg_pnn_k = [0 for _ in range(num_product)], [0 for _ in range(num_product)]
                                    avg_pro_k, avg_bud_k = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

                                    eva_main = Evaluation(graph_dict, seed_cost_dict, product_list, pps, wpiwp)
                                    iniW = IniWallet(data_set_name, product_name, distribution_type)
                                    wallet_list = iniW.getWalletList()
                                    personal_prob_list = eva_main.setPersonalProbList(wallet_list)
                                    for sample_count, sample_seed_set in enumerate(seed_set_sequence[bud - 1]):
                                        print('@ mngaprpwic evaluation @ data_set_name = ' + data_set_name + '_' + cas_model + ', dis = ' + str(distribution_type) + ', wpiwp = ' + str(wpiwp) +
                                              ', product_name = ' + product_name + ', budget = ' + str(bud) + ', pps = ' + str(pps) + ', sample_count = ' + str(sample_count))
                                        sample_pro_acc, sample_bud_acc = 0.0, 0.0
                                        sample_sn_k_acc, sample_pnn_k_acc = [0.0 for _ in range(num_product)], [0 for _ in range(num_product)]
                                        sample_pro_k_acc, sample_bud_k_acc = [0.0 for _ in range(num_product)], [0.0 for _ in range(num_product)]

                                        for _ in range(eva_monte_carlo):
                                            pro, pro_k_list, pnn_k_list = eva_main.getSeedSetProfit(sample_seed_set, copy.deepcopy(wallet_list), copy.deepcopy(personal_prob_list))
                                            sample_pro_acc += pro
                                            for kk in range(num_product):
                                                sample_pro_k_acc[kk] += pro_k_list[kk]
                                                sample_pnn_k_acc[kk] += pnn_k_list[kk]
                                        sample_pro_acc = round(sample_pro_acc / eva_monte_carlo, 4)
                                        for kk in range(num_product):
                                            sample_pro_k_acc[kk] = round(sample_pro_k_acc[kk] / eva_monte_carlo, 4)
                                            sample_pnn_k_acc[kk] = round(sample_pnn_k_acc[kk] / eva_monte_carlo, 2)
                                            sample_sn_k_acc[kk] = len(sample_seed_set[kk])
                                            for sample_seed in sample_seed_set[kk]:
                                                sample_bud_acc += seed_cost_dict[sample_seed]
                                                sample_bud_k_acc[kk] += seed_cost_dict[sample_seed]
                                                sample_bud_acc = round(sample_bud_acc, 2)
                                                sample_bud_k_acc[kk] = round(sample_bud_k_acc[kk], 2)

                                        result[pps - 1].append([sample_pro_acc, sample_bud_acc, sample_sn_k_acc, sample_pnn_k_acc, sample_seed_set])
                                        avg_pro += sample_pro_acc
                                        avg_bud += sample_bud_acc
                                        for kk in range(num_product):
                                            avg_sn_k[kk] += sample_sn_k_acc[kk]
                                            avg_pnn_k[kk] += sample_pnn_k_acc[kk]
                                            avg_pro_k[kk] += sample_pro_k_acc[kk]
                                            avg_bud_k[kk] += sample_bud_k_acc[kk]

                                        print('eva_time = ' + str(round(time.time() - eva_start_time, 2)) + 'sec')
                                        print(result[pps - 1][sample_count])
                                        print('avg_profit = ' + str(round(avg_pro / (sample_count + 1), 4)) + ', avg_budget = ' + str(round(avg_bud / (sample_count + 1), 4)))
                                        print('------------------------------------------')

                                    avg_pro = round(avg_pro / sample_number, 4)
                                    avg_bud = round(avg_bud / sample_number, 2)
                                    for kk in range(num_product):
                                        avg_sn_k[kk] = round(avg_sn_k[kk] / sample_number, 2)
                                        avg_pnn_k[kk] = round(avg_pnn_k[kk] / sample_number, 2)
                                        avg_pro_k[kk] = round(avg_pro_k[kk] / sample_number, 4)
                                        avg_bud_k[kk] = round(avg_bud_k[kk] / sample_number, 2)

                                    total_time = round(sum(ss_time_sequence[bud - 1]), 2)
                                    path1 = 'result/mngaprpwic_pps' + str(pps) + '_dis' + str(distribution_type) + '_wpiwp' * wpiwp
                                    if not os.path.isdir(path1):
                                        os.mkdir(path1)
                                    path = 'result/mngaprpwic_pps' + str(pps) + '_dis' + str(distribution_type) + '_wpiwp' * wpiwp + '/' + data_set_name + '_' + cas_model + '_' + product_name
                                    if not os.path.isdir(path):
                                        os.mkdir(path)
                                    fw = open(path + '/b' + str(bud) + '_i' + str(sample_number) + '.txt', 'w')
                                    fw.write('mngaprpwic, pp_strategy = ' + str(pps) + ', total_budget = ' + str(bud) + ', dis = ' + str(distribution_type) + ', wpiwp = ' + str(wpiwp) + '\n' +
                                             'data_set_name = ' + data_set_name + '_' + cas_model + ', product_name = ' + product_name + '\n' +
                                             'total_budget = ' + str(bud) + ', sample_count = ' + str(sample_number) + '\n' +
                                             'avg_profit = ' + str(avg_pro) + ', avg_budget = ' + str(avg_bud) + '\n' +
                                             'total_time = ' + str(total_time) + ', avg_time = ' + str(round(total_time / sample_number, 4)) + '\n')
                                    fw.write('\nprofit_ratio =')
                                    for kk in range(num_product):
                                        fw.write(' ' + str(avg_pro_k[kk]))
                                    fw.write('\nbudget_ratio =')
                                    for kk in range(num_product):
                                        fw.write(' ' + str(avg_bud_k[kk]))
                                    fw.write('\nseed_number =')
                                    for kk in range(num_product):
                                        fw.write(' ' + str(avg_sn_k[kk]))
                                    fw.write('\ncustomer_number =')
                                    for kk in range(num_product):
                                        fw.write(' ' + str(avg_pnn_k[kk]))
                                    fw.write('\n')

                                    for t, r in enumerate(result[pps - 1]):
                                        fw.write('\n' + str(t) + '\t' + str(round(r[0], 4)) + '\t' + str(round(r[1], 4)) + '\t' + str(r[2]) + '\t' + str(r[3]) + '\t' + str(r[4]))
                                    fw.close()
