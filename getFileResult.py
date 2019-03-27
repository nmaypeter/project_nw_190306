import os

data_setting_seq = [1]
cm_seq = [1, 2]
# model is optional
model_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
pps_seq = [1, 2, 3]
wpiwp_seq = [bool(0), bool(1)]
prod_setting_seq, prod_setting2_seq = [1, 2], [1, 2, 3]
dis_seq = [1, 2]

for data_setting in data_setting_seq:
    data_set_name = 'email_undirected' * (data_setting == 1) + 'dnc_email_directed' * (data_setting == 2) + 'email_Eu_core_directed' * (data_setting == 3) + \
                    'WikiVote_directed' * (data_setting == 4) + 'NetPHY_undirected' * (data_setting == 5)
    for cm in cm_seq:
        cas_model = 'ic' * (cm == 1) + 'wc' * (cm == 2)
        for m in model_seq:
            model_name = 'mngic' * (m == 1) + 'mhdic' * (m == 2) + 'mric' * (m == 3) + 'mpmisic' * (m == 4) + \
                         'mngric' * (m == 5) + 'mngpwic' * (m == 6) + 'mngrpwic' * (m == 7) + \
                         'mhedic' * (m == 8) + 'mhdpwic' * (m == 9) + 'mhedpwic' * (m == 10) + \
                         'mngapic' * (m == 11) + 'mngapric' * (m == 12) + 'mngappwic' * (m == 13) + 'mngaprpwic' * (m == 14) + 'mpmisapic' * (m == 15) + '_pps'
            for pps in pps_seq:
                for wpiwp in wpiwp_seq:
                    for prod_setting in prod_setting_seq:
                        for prod_setting2 in prod_setting2_seq:
                            for dis in dis_seq:
                                product_name = 'r1p3n' + str(prod_setting) + 'a' * (prod_setting2 == 2) + 'b' * (prod_setting2 == 3)
                                total_budget = 10

                                num_ratio, num_price = int(list(product_name)[list(product_name).index('r') + 1]), int(list(product_name)[list(product_name).index('p') + 1])
                                num_product = num_ratio * num_price

                                profit, cost = [], []
                                time_avg, time_total = [], []

                                ratio_profit, ratio_cost = [[] for _ in range(num_product)], [[] for _ in range(num_product)]
                                number_seed, number_an = [[] for _ in range(num_product)], [[] for _ in range(num_product)]

                                for bud in range(1, total_budget + 1):
                                    try:
                                        result_name = ''
                                        if m > 10:
                                            result_name = 'result/' + model_name + str(pps) + '_dis' + str(dis) + '_wpiwp' * wpiwp + '/' + \
                                                          data_set_name + '_' + cas_model + '_' + product_name + '/' + 'b' + str(bud) + '_i1.txt'
                                        else:
                                            result_name = 'result/' + model_name + str(pps) + '_dis' + str(dis) + '_wpiwp' * wpiwp + '/' + \
                                                          data_set_name + '_' + cas_model + '_' + product_name + '/' + 'b' + str(bud) + '_i10.txt'
                                        print(result_name)

                                        with open(result_name) as f:
                                            for lnum, line in enumerate(f):
                                                if lnum <= 2 or lnum == 5:
                                                    continue
                                                elif lnum == 3:
                                                    (l) = line.split()
                                                    profit.append(l[2].rstrip(','))
                                                    cost.append(l[-1])
                                                elif lnum == 4:
                                                    (l) = line.split()
                                                    time_total.append(l[2].rstrip(','))
                                                    time_avg.append(l[-1])
                                                elif lnum == 6:
                                                    (l) = line.split()
                                                    for nl in range(2, len(l)):
                                                        ratio_profit[nl-2].append(l[nl])
                                                elif lnum == 7:
                                                    (l) = line.split()
                                                    for nl in range(2, len(l)):
                                                        ratio_cost[nl - 2].append(l[nl])
                                                elif lnum == 8:
                                                    (l) = line.split()
                                                    for nl in range(2, len(l)):
                                                        number_seed[nl-2].append(l[nl])
                                                elif lnum == 9:
                                                    (l) = line.split()
                                                    for nl in range(2, len(l)):
                                                        number_an[nl - 2].append(l[nl])
                                                else:
                                                    break
                                        f.close()
                                    except FileNotFoundError:
                                        profit.append('')
                                        cost.append('')
                                        time_total.append('')
                                        time_avg.append('')
                                        for num in range(num_product):
                                            ratio_profit[num].append('')
                                            ratio_cost[num].append('')
                                            number_seed[num].append('')
                                            number_an[num].append('')
                                        continue

                                    path1 = 'result/r_' + data_set_name
                                    if not os.path.isdir(path1):
                                        os.mkdir(path1)
                                    path2 = 'result/r_' + data_set_name + '_' + cas_model + '/' + model_name + str(pps) + '_dis' + str(dis) + '_wpiwp' * wpiwp
                                    if not os.path.isdir(path2):
                                        os.mkdir(path2)
                                    path = 'result/r_' + data_set_name + '_' + cas_model + '/' + model_name + str(pps) + '_dis' + str(dis) + '_wpiwp' * wpiwp + '/' + \
                                           model_name + str(pps) + '_wpiwp' * wpiwp + '_' + product_name
                                    if not os.path.isdir(path):
                                        os.mkdir(path)

                                    fw = open(path + '/1profit.txt', 'w')
                                    for line in profit:
                                        fw.write(str(line) + '\t')
                                    fw.close()
                                    fw = open(path + '/2cost.txt', 'w')
                                    for line in cost:
                                        fw.write(str(line) + '\t')
                                    fw.close()
                                    fw = open(path + '/3time_avg.txt', 'w')
                                    for line in time_avg:
                                        fw.write(str(line) + '\t')
                                    fw.close()
                                    fw = open(path + '/4time_total.txt', 'w')
                                    for line in time_total:
                                        fw.write(str(line) + '\t')
                                    fw.close()
                                    fw = open(path + '/5ratio_profit.txt', 'w')
                                    for line in ratio_profit:
                                        for l in line:
                                            fw.write(str(l) + '\t')
                                        fw.write('\n')
                                    fw.close()
                                    fw = open(path + '/6ratio_cost.txt', 'w')
                                    for line in ratio_cost:
                                        for l in line:
                                            fw.write(str(l) + '\t')
                                        fw.write('\n')
                                    fw.close()
                                    fw = open(path + '/7number_pn.txt', 'w')
                                    for line in number_an:
                                        for l in line:
                                            fw.write(str(l) + '\t')
                                        fw.write('\n')
                                    fw.close()
                                    fw = open(path + '/8number_seed.txt', 'w')
                                    for line in number_seed:
                                        for l in line:
                                            fw.write(str(l) + '\t')
                                        fw.write('\n')
                                    fw.close()
