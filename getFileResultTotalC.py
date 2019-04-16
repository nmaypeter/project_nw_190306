data_setting_seq = [1, 2]
cm_seq = [1, 2]
model_kinds = 14
pps_seq = [1, 2, 3]
wpiwp_seq = [bool(0), bool(1)]
prod_setting_seq, prod_setting2_seq = [1, 2], [1, 2, 3]
dis_seq = [1, 2]

for data_setting in data_setting_seq:
    data_set_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                    'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
    for cm in cm_seq:
        cas_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        for pps in pps_seq:
            for dis in dis_seq:
                profit = []
                for prod_setting in prod_setting_seq:
                    for prod_setting2 in prod_setting2_seq:
                        product_name = 'r1p3n' + str(prod_setting) + 'a' * (prod_setting2 == 2) + 'b' * (prod_setting2 == 3)
                        for wpiwp in wpiwp_seq:
                            for m in range(1, model_kinds + 1):
                                model_name = 'mngic' * (m == 1) + 'mhdic' * (m == 2) + 'mric' * (m == 3) + 'mpmisic' * (m == 4) + \
                                             'mngric' * (m == 5) + 'mngpwic' * (m == 6) + 'mngrpwic' * (m == 7) + \
                                             'mhedic' * (m == 8) + 'mhdpwic' * (m == 9) + 'mhedpwic' * (m == 10) + \
                                             'mngapic' * (m == 11) + 'mngapric' * (m == 12) + 'mngappwic' * (m == 13) + 'mngaprpwic' * (m == 14) + '_pps'

                                try:
                                    result_name = 'result/r_' + data_set_name + '_' + cas_model + '/' + model_name + str(pps) + '_dis' + str(dis) + '_wpiwp' * wpiwp \
                                                  + '/' + model_name + str(pps) + '_wpiwp' * wpiwp + '_' + product_name + '/1profit.txt'
                                    print(result_name)

                                    with open(result_name) as f:
                                        for lnum, line in enumerate(f):
                                            if lnum == 0:
                                                profit.append(line)
                                            else:
                                                break
                                    f.close()
                                except FileNotFoundError:
                                    profit.append('')
                                    continue

                fw = open('result/r_' + data_set_name + '_' + cas_model + '/dis' + str(dis) + '_pps' + str(pps) + '_comparison_total_profit.txt', 'w')
                for lnum, line in enumerate(profit):
                    if lnum % (model_kinds * 2) == 0 and lnum != 0:
                        fw.write('\n' * 3)
                    fw.write(str(line) + '\n')
                fw.close()
