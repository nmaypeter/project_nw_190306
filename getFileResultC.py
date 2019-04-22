model_seq = [['mngic', 'mhdic', 'mric', 'mpmisic'],
             ['mngic', 'mngric', 'mngpwic', 'mngrpwic', 'mngsric', 'mngsrpwic'],
             ['mngapic', 'mngapric', 'mngappwic', 'mngaprpwic', 'mngapsric', 'mngapsrpwic'],
             ['mhdic', 'mhedic', 'mhdpwic', 'mhedpwic']]
data_setting_seq = [1, 2]
cm_seq = [1, 2]
pps_seq = [1, 2, 3]
wpiwp_seq = [bool(0), bool(1)]
prod_setting_seq, prod_setting2_seq = [1, 2], [1, 2, 3]
dis_seq = [1, 2]

for data_setting in data_setting_seq:
    data_set_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                    'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
    for cm in cm_seq:
        cas_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        for dis in dis_seq:
            profit = []
            for prod_setting in prod_setting_seq:
                for prod_setting2 in prod_setting2_seq:
                    product_name = 'r1p3n' + str(prod_setting) + 'a' * (prod_setting2 == 2) + 'b' * (prod_setting2 == 3)
                    for wpiwp in wpiwp_seq:
                        for ms in model_seq:
                            for m in ms:
                                model_name = m + '_pps'
                                for pps in pps_seq:
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

                            fw = ''
                            if model_seq.index(ms) == 0:
                                fw = open('result/r_' + data_set_name + '_' + cas_model + '/dis' + str(dis) + '_comparison_profit.txt', 'w')
                            elif model_seq.index(ms) == 1:
                                fw = open('result/r_' + data_set_name + '_' + cas_model + '/dis' + str(dis) + '_comparison_ng_profit.txt', 'w')
                            elif model_seq.index(ms) == 2:
                                fw = open('result/r_' + data_set_name + '_' + cas_model + '/dis' + str(dis) + '_comparison_ap_profit.txt', 'w')
                            elif model_seq.index(ms) == 3:
                                fw = open('result/r_' + data_set_name + '_' + cas_model + '/dis' + str(dis) + '_comparison_hd_profit.txt', 'w')
                            for lnum, line in enumerate(profit):
                                if lnum % (len(ms) * 3) == 0 and lnum != 0:
                                    fw.write('\n')
                                if lnum % (len(ms) * 6) == 0 and lnum != 0:
                                    fw.write('\n')
                                fw.write(str(line) + '\n')
                            fw.close()
