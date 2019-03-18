data_setting_seq = [1]
model_kinds = 10
pps_seq = [1, 2, 3]
wpiwp_seq = [bool(0), bool(1)]
prod_setting_seq, prod_setting2_seq = [1, 2], [1, 2, 3]
dis_seq = [1, 2]
total_budget = 10

for bud in range(1, total_budget + 1):
    for data_setting in data_setting_seq:
        data_set_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                        'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
        profit = []
        for wpiwp in wpiwp_seq:
            for dis in dis_seq:
                for pps in pps_seq:
                    for prod_setting in prod_setting_seq:
                        for prod_setting2 in prod_setting2_seq:
                            product_name = 'r1p3n' + str(prod_setting) + 'a' * (prod_setting2 == 2) + 'b' * (prod_setting2 == 3)
                            model_str = ''
                            for m in range(1, model_kinds + 1):
                                model_name = 'mngic' * (m == 1) + 'mhdic' * (m == 2) + 'mric' * (m == 3) + 'mpmisic' * (m == 4) + \
                                             'mngric' * (m == 5) + 'mngpwic' * (m == 6) + 'mngrpwic' * (m == 7) + \
                                             'mhedic' * (m == 8) + 'mhdpwic' * (m == 9) + 'mhedpwic' * (m == 10) + '_pps'
                                try:
                                    result_name = 'result/' + model_name + str(pps) + '_dis' + str(dis) + '_wpiwp' * wpiwp + '/' + \
                                                  data_set_name + '_' + product_name + '/' + 'b' + str(bud) + '_i10.txt'
                                    print(result_name)

                                    with open(result_name) as f:
                                        for lnum, line in enumerate(f):
                                            if lnum == 3:
                                                (l) = line.split()
                                                model_str += (l[2].rstrip(',')) + '\t'
                                                break
                                    f.close()
                                except FileNotFoundError:
                                    model_str += '\t'
                            profit.append(model_str)

        fw = open('result/r_' + data_set_name + '/comparison_total_profit_analysis_b' + str(bud) + '.txt', 'w')
        for lnum, line in enumerate(profit):
            if lnum % (len(dis_seq) * len(pps_seq) * len(prod_setting_seq) * len(prod_setting2_seq)) == 0 and lnum != 0:
                fw.write('\n')
            fw.write(str(line) + '\n')
        fw.close()