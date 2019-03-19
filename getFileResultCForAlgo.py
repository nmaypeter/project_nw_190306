ctl_seq = [0, 1, 2]
data_setting_seq = [1]
model_kinds = 4
pps_seq = [1, 2, 3]
wpiwp_seq = [bool(0), bool(1)]
prod_setting_seq, prod_setting2_seq = [1, 2], [1, 2, 3]
dis_seq = [1, 2]

for ctl in ctl_seq:
    for data_setting in data_setting_seq:
        data_set_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                        'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
        for pps in pps_seq:
            for dis in dis_seq:
                profit = []
                for prod_setting in prod_setting_seq:
                    for prod_setting2 in prod_setting2_seq:
                        product_name = 'r1p3n' + str(prod_setting) + 'a' * (prod_setting2 == 2) + 'b' * (prod_setting2 == 3)
                        for wpiwp in wpiwp_seq:
                            for m in range(1, model_kinds + 1):
                                model_name = ''
                                if ctl == 0:
                                    model_name = 'mngic' * (m == 1) + 'mhdic' * (m == 2) + 'mric' * (m == 3) + 'mpmisic' * (m == 4) + '_pps'
                                elif ctl == 1:
                                    model_name = 'mngic' * (m == 1) + 'mngric' * (m == 2) + 'mngpwic' * (m == 3) + 'mngrpwic' * (m == 4) + '_pps'
                                elif ctl == 2:
                                    model_name = 'mhdic' * (m == 1) + 'mhedic' * (m == 2) + 'mhdpwic' * (m == 3) + 'mhedpwic' * (m == 4) + '_pps'

                                try:
                                    result_name = 'result/r_' + data_set_name + '/' + model_name + str(pps) + '_dis' + str(dis) + '_wpiwp' * wpiwp \
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

                fw = open('result/r_' + data_set_name + '/dis' + str(dis) + '_pps' + str(pps) + '_comparison_profit.txt', 'w')
                if ctl == 1:
                    fw = open('result/r_' + data_set_name + '/dis' + str(dis) + '_pps' + str(pps) + '_comparison_ng_profit.txt', 'w')
                elif ctl == 2:
                    fw = open('result/r_' + data_set_name + '/dis' + str(dis) + '_pps' + str(pps) + '_comparison_hd_profit.txt', 'w')
                for lnum, line in enumerate(profit):
                    if lnum % (model_kinds * 2) == 0 and lnum != 0:
                        fw.write('\n' * 9)
                    fw.write(str(line) + '\n')
                fw.close()