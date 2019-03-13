import os

data_setting_seq = [1]
# model is optional
model_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pps_seq = [1, 2, 3]
wpiwp_seq = [bool(0), bool(1)]
prod_setting_seq, prod_setting2_seq = [1, 2], [1, 2, 3]
dis_seq = [1, 2]
for data_setting in data_setting_seq:
    data_set_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                    'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
    for m in model_seq:
        model_name = 'mngic' * (m == 1) + 'mhdic' * (m == 2) + 'mric' * (m == 3) + 'mpmisic' * (m == 4) + \
                     'mngscsic' * (m == 5) + 'mngric' * (m == 6) + 'mngpwic' * (m == 7) + \
                     'mhedic' * (m == 8) + 'mhdpwic' * (m == 9) + 'mhedpwic' * (m == 10) + '_pps'
        for pps in pps_seq:
            profit, cost, time_avg, time_total = [], [], [], []
            ratio_profit, ratio_cost, number_an, number_seed = [], [], [], []
            for dis in dis_seq:
                for prod_setting in prod_setting_seq:
                    for wpiwp in wpiwp_seq:
                        for prod_setting2 in prod_setting2_seq:
                            try:
                                product_name = 'r1p3n' + str(prod_setting) + 'a' * (prod_setting2 == 2) + 'b' * (prod_setting2 == 3)
                                result_name = 'result/r_' + data_set_name + '/' + model_name + str(pps) + '_dis' + str(dis) + '_wpiwp' * wpiwp \
                                              + '/' + model_name + str(pps) + '_wpiwp' * wpiwp + '_' + product_name

                                with open(result_name + '/1profit.txt') as f:
                                    for line in f:
                                        profit.append(line)
                                f.close()
                                with open(result_name + '/2cost.txt') as f:
                                    for line in f:
                                        cost.append(line)
                                f.close()
                                with open(result_name + '/3time_avg.txt') as f:
                                    for line in f:
                                        time_avg.append(line)
                                f.close()
                                with open(result_name + '/4time_total.txt') as f:
                                    for line in f:
                                        time_total.append(line)
                                f.close()
                            except FileNotFoundError:
                                profit.append('')
                                cost.append('')
                                time_total.append('')
                                time_avg.append('')
                                continue

                    for prod_setting2 in [1, 2, 3]:
                        product_name = 'r1p3n' + str(prod_setting) + 'a' * (prod_setting2 == 2) + 'b' * (prod_setting2 == 3)
                        num_ratio, num_price = int(list(product_name)[list(product_name).index('r') + 1]), int(list(product_name)[list(product_name).index('p') + 1])
                        num_product = num_ratio * num_price
                        for wpiwp in [bool(0), bool(1)]:
                            try:
                                result_name = 'result/r_' + data_set_name + '/' + model_name + str(pps) + '_dis' + str(dis) + '_wpiwp' * wpiwp \
                                              + '/' + model_name + str(pps) + '_wpiwp' * wpiwp + '_' + product_name

                                with open(result_name + '/5ratio_profit.txt') as f:
                                    for line in f:
                                        ratio_profit.append(line)
                                f.close()
                                with open(result_name + '/6ratio_cost.txt') as f:
                                    for line in f:
                                        ratio_cost.append(line)
                                f.close()
                                with open(result_name + '/7number_pn.txt') as f:
                                    for line in f:
                                        number_an.append(line)
                                f.close()
                                with open(result_name + '/8number_seed.txt') as f:
                                    for line in f:
                                        number_seed.append(line)
                                f.close()
                            except FileNotFoundError:
                                for num in range(num_product):
                                    ratio_profit.append('\n')
                                    ratio_cost.append('\n')
                                    number_seed.append('\n')
                                    number_an.append('\n')
                                continue

                path1 = 'result/r_' + data_set_name + '/r_dis' + str(dis)
                if not os.path.isdir(path1):
                    os.mkdir(path1)
                path = 'result/r_' + data_set_name + '/r_dis' + str(dis) + '/' + model_name + str(pps)
                fw = open(path + '_1profit.txt', 'w')
                for lnum, line in enumerate(profit):
                    fw.write(str(line) + '\n')
                    if lnum == 5:
                        fw.write('\n' * 10)
                fw.close()
                fw = open(path + '_2cost.txt', 'w')
                for lnum, line in enumerate(cost):
                    fw.write(str(line) + '\n')
                    if lnum == 5:
                        fw.write('\n' * 10)
                fw.close()
                fw = open(path + '_3time_avg.txt', 'w')
                for lnum, line in enumerate(time_avg):
                    fw.write(str(line) + '\n')
                    if lnum == 5:
                        fw.write('\n' * 10)
                fw.close()
                fw = open(path + '_4time_total.txt', 'w')
                for lnum, line in enumerate(time_total):
                    fw.write(str(line) + '\n')
                    if lnum == 5:
                        fw.write('\n' * 10)
                fw.close()

                fw = open(path + '_5ratio_profit.txt', 'w')
                for lnum, line in enumerate(ratio_profit):
                    if lnum % 6 == 0 and lnum != 0:
                        fw.write('\n' * 9)
                    fw.write(str(line))
                fw.close()
                fw = open(path + '_6ratio_cost.txt', 'w')
                for lnum, line in enumerate(ratio_cost):
                    if lnum % 6 == 0 and lnum != 0:
                        fw.write('\n' * 9)
                    fw.write(str(line))
                fw.close()
                fw = open(path + '_7number_pn.txt', 'w')
                for lnum, line in enumerate(number_an):
                    if lnum % 6 == 0 and lnum != 0:
                        fw.write('\n' * 9)
                    fw.write(str(line))
                fw.close()
                fw = open(path + '_8number_seed.txt', 'w')
                for lnum, line in enumerate(number_seed):
                    if lnum % 6 == 0 and lnum != 0:
                        fw.write('\n' * 9)
                    fw.write(str(line))
                fw.close()
