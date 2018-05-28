import numpy as np
import pandas as pd
from itertools import permutations
from sklearn import decomposition
from pylab import *
from time import time

# Paramters
iterations = 5000
family_size = 500
train_split = 0.2
drop_variable = 'Scanner'
use_binary = True
decomp_mode = 'ica'
use_big_data = False
use_zero_threshold = True
n_components = 200

def get_new_chromosom(chrom_size, rate=20, binary=True, zero_threshold=True):
    if binary:
        chrom = (np.random.randint(
            1, 100, size=chrom_size) < rate).astype('bool')
    else:
        chrom = np.random.rand(chrom_size)

    if zero_threshold:
        threshold = 0
    else:
        threshold = np.random.randn()
    return threshold, chrom


def compute_fitness(data, labels, chromosom):

    threshold, chrom = chromosom

    # Make sure that chromosom is not all zero
    if chrom.sum() == 0:
        return 0
    else:
        prediction = np.sum(data * chrom, 1) >= threshold
        target = labels == 1

    TP = np.sum(prediction[target])
    TN = np.sum(np.invert(prediction[np.invert(target)]))
    FP = np.sum(prediction[np.invert(target)])
    FN = np.sum(np.invert(prediction[target]))
    P = TP + FN
    N = TN + FP

    sensitivity = 1. * TP / (TP + FN)
    specificity = 1. * TN / (FP + TN)
    precision = 1. * TP / (TP + FP)
    accuracy = 1. * (TP + TN) / (P + N)

    pac_value = np.mean([sensitivity, specificity])

    return pac_value


def judge_family(data, labels, family, selecter):
    return [[compute_fitness(data[selecter],
                             labels[selecter],
                             chromosom)
             for chromosom in family],
            [compute_fitness(data[np.invert(selecter)],
                             labels[np.invert(selecter)],
                             chromosom)
             for chromosom in family]]


def run_evolution(iterations, family_size, use_big_data, train_split,
                  use_binary, use_zero_threshold, drop_variable, decomp_mode,
                  pool_info, n_components):

    # Import metadata to know who is control and who is patient
    df = pd.read_csv('data/PAC2018_Covariates_pooling_red%s.csv' % pool_info, index_col=0)

    if use_group == 1:
        df = df[df.Scanner == 1]
        postfix = 'scanner1'
    elif use_group == 0:
        df = df[df.Scanner != 1]
        postfix = 'scanner23'
    elif use_group == -1:
        df = df[df.Scanner != 0]
        postfix = 'scanner0'

    if drop_variable != '':
        df = df.drop(drop_variable, 1)
    labels = np.array(df['Label']) - 1
    sub = df.index

    # Get data in right form
    data = df.drop(['Label'], 1)

    # Compute the factor analysis
    if decomp_mode == 'faa':
        faa = decomposition.FactorAnalysis(n_components=n_components)
        faa.fit(data)
        data = faa.transform(data)

    elif decomp_mode == 'ica':
        ica = decomposition.FastICA(n_components=n_components)
        ica.fit(data)
        data = ica.transform(data)

    if use_big_data:
        new_data = np.copy(data)

        for i in range(data.shape[1]):
            temp = data * data[:, i][:, None]
            new_data = np.hstack((new_data, temp))
        data = np.copy(new_data)

    # Import test data for prediction
    df_test = pd.read_csv('data/PAC2018_Covariates_Test_pooling_red%s.csv' % pool_info, index_col=0)

    if use_group == 1:
        df_test = df_test[df_test.Scanner == 1]
    elif use_group == 0:
        df_test = df_test[df_test.Scanner != 1]
    elif use_group == -1:
        df_test = df_test[df_test.Scanner != 0]

    if drop_variable != '':
        df_test = df_test.drop(drop_variable, 1)

    # Get data in right form
    data_test = df_test.drop(['Label'], 1)

    if decomp_mode == 'faa':
        data_test = faa.transform(data_test)

    elif decomp_mode == 'ica':
        data_test = ica.transform(data_test)

    if use_big_data:
        new_data = np.copy(data_test)

        for i in range(data_test.shape[1]):
            temp = data_test * data_test[:, i][:, None]
            new_data = np.hstack((new_data, temp))
        data_test = np.copy(new_data)

    # Specify chromosoms
    chrom_size = data.shape[1]

    # Create new family
    family = [get_new_chromosom(chrom_size, binary=use_binary,
                                zero_threshold=use_zero_threshold)
              for x in range(family_size)]

    result_train = []
    result_test = []
    result_thresh = []
    evolution = []

    for k in range(iterations):

        if k % 50 == 0:

            # Balance dataset and create selecter
            max_label_size = np.min([np.sum(lab == labels)
                                     for lab in np.unique(labels)])

            labels_1 = np.where(labels == 0)[0]
            np.random.shuffle(labels_1)
            labels_1 = labels_1[:max_label_size]

            labels_2 = np.where(labels == 1)[0]
            np.random.shuffle(labels_2)
            labels_2 = labels_2[:max_label_size]

            # Balance dataset
            new_data_id = np.hstack((labels_1, labels_2))
            np.random.shuffle(new_data_id)
            data_balanced = data[new_data_id]
            labels_balanced = labels[new_data_id]

            # Create selecter
            test_size = int(((100 - (100 * train_split)) / 100.) * max_label_size)
            selecter = np.zeros(len(labels_balanced))
            selecter[:test_size] = 1
            selecter[max_label_size:max_label_size + test_size] = 1
            selecter = selecter.astype('bool')

        # Calculating fittnes using multiprocessing (=parallel)
        fitness = judge_family(
            data_balanced, labels_balanced, family, selecter)
        fit_train = np.array(fitness[0])
        fit_test = np.array(fitness[1])

        good_parents = fit_train.argsort()[-32:]

        # Save best chromosom
        evolution.append([family[good_parents[-1]],
                          family[good_parents[-2]],
                          family[good_parents[-3]]])

        # Get clear File identifier
        file_id = 'iter_%05d_family_%04d_bin_%s_zeroThresh_%s_group_%s_comp_%s%d' % (
            iterations, family_size, use_binary, use_zero_threshold, postfix, decomp_mode, n_components)


        # Create new family
        new_family = [family[g] for g in good_parents]

        # Create childrens
        for c in permutations(range(8), 2):

            new_child = np.zeros(chrom_size)
            if use_binary:
                new_child = new_child.astype('bool')
            half_id = int(chrom_size / 2)
            new_child[:half_id] = new_family[c[0]][1][:half_id]
            new_child[half_id:] = new_family[c[1]][1][half_id:]
            new_threshold = new_family[c[0]][0]
            new_family.append((new_threshold, new_child))

        # Vary threshold in good parents (if not zero threshold)
        if not use_zero_threshold:
            for f in [family[g] for g in good_parents]:
                new_threshold = np.random.randn()
                new_family.append((new_threshold, f[1]))

        # Create possible mutations for each family member
        family_length = len(new_family)
        for i in range(family_length):
            for j in [[0, 33], [33, 67], [67, 100]]:
                element = new_family[i]
                mut_rate = np.random.randint(j[0], j[1])
                mutation = get_new_chromosom(
                    chrom_size, rate=np.random.randint(1, 100),
                    binary=use_binary, zero_threshold=use_zero_threshold)

                if np.random.random() * 100 <= mut_rate:
                    mut_threshold = mutation[0]
                else:
                    mut_threshold = element[0]

                mutant = element[1].copy()

                mut_hit = (np.random.randint(1, 100, size=chrom_size)
                           < mut_rate).astype('bool')

                mutant[mut_hit] = mutation[1][mut_hit]
                new_family.append((mut_threshold, mutant))

        # Find duplicates
        analysis_format = [[float(f[0])] + list(f[1].astype('float'))
                           for f in new_family]
        a = np.asarray(analysis_format)
        b = np.ascontiguousarray(a).view(
            np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
        a = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])

        if use_binary:
            new_family = [(newfam[0], newfam[1:].astype('bool'))
                          for newfam in a]
        else:
            new_family = [(newfam[0], newfam[1:]) for newfam in a]

        # Add new chromosoms
        for j in [[0, 20], [20, 40], [40, 60], [60, 80], [80, 100]]:
            for i in range(10):
                mut_rate = np.random.randint(j[0], j[1])
                new_family.append(get_new_chromosom(
                    chrom_size, rate=mut_rate, binary=use_binary,
                    zero_threshold=use_zero_threshold))

        # Add rest of chromosoms
        for j in range(family_size - len(new_family)):
            new_family.append(get_new_chromosom(
                chrom_size, rate=np.random.randint(1, 100),
                binary=use_binary, zero_threshold=use_zero_threshold))

        # Reset the family
        family = new_family

        acc_train = np.round(fit_train.max() * 100, 4)
        acc_test = np.round(fit_test.max() * 100, 4)
        result_train.append(acc_train)
        result_test.append(acc_test)
        acc_threshold = round(new_family[0][0], 3)
        result_thresh.append(acc_threshold)

        print(k, acc_train, acc_test, acc_threshold)

        acc_both = [acc_train, acc_test]
        if np.mean(acc_both) >= 70 and np.min(acc_both) >= 67.5:
            strong_thresh = evolution[-1][0][0]
            strong_chrom = evolution[-1][0][1]

            predict = np.sum(data_test * strong_chrom, 1) >= strong_thresh
            predict = (predict + 1).tolist()
            predict = [[np.mean(acc_both)] + acc_both + predict]

            np.savetxt('results/evolution_pooling/strong_%s_%s_%s.txt' % (
                pool_info, file_id, str(time())), predict, fmt='%f', delimiter=',')

    title_text = ' Acc = %s - Iter: %04d - Family: %04d - Big: %s - ' % (
        round(acc_train, 2), iterations, family_size, use_big_data)
    title_text += 'Binary: %s - ZeroThresh: %s - Group: %s' % (
        use_binary, use_zero_threshold, postfix)
    if drop_variable != '':
        title_text += ' - Dropped: %s' % drop_variable

    title_text += ' - Comp: %s' % decomp_mode

    result_mean = (np.array(result_train) + np.array(result_test)) / 2

    figure(figsize=(16, 6))
    plot(result_train)
    plot(result_test)
    plot(result_mean)
    plot(np.array(result_thresh) + 60)
    legend(['Train [~%0.1f]' % np.mean(result_train[200:]),
            'Test [~%0.1f]' % np.mean(result_test[200:]),
            'Average [~%0.1f]' % np.mean(result_mean[200:]),
            'Threshold [+60]'])
    title('Fitness:%s - Threshold at %f' % (title_text, result_thresh[-1]))
    xlabel('Generation')
    ylabel('Accuracy [%]')
    tight_layout()
    savefig('results/evolution_pooling/fitness_%s_%s.png' % (pool_info, file_id))
    close()

    comp_name = ['comp_%03d' % (r + 1) for r in range(data.shape[1])]

    evolution = np.array([np.array([[float(f[0])] + list(f[1].astype('float'))
                                    for f in ev]) for ev in evolution])
    evolutionRGB = np.rollaxis(np.rollaxis(evolution, 2), 1).astype('float32')
    figure(figsize=(16, 8))
    imshow(evolutionRGB, aspect='auto')
    title('Chromosom:%s - Threshold at %f' % (title_text, result_thresh[-1]))
    ylabel('Generation')
    xticks(range(chrom_size + 1), ['Threshold'] + comp_name,
           rotation='vertical')
    subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.15)
    savefig('results/evolution_pooling/chromosom_%s_%s.png' % (pool_info, file_id))
    close()

    family = np.array([[float(f[0])] + list(f[1].astype('float'))
                       for f in family])
    figure(figsize=(16, 8))
    imshow(family, aspect='auto')
    title('Final Family:%s - Threshold at %f' %
          (title_text, result_thresh[-1]))
    ylabel('Generation')
    xticks(range(chrom_size + 1), ['Threshold'] + comp_name,
           rotation='vertical')
    subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.15)
    savefig('results/evolution_pooling/family_%s_%s.png' % (pool_info, file_id))
    close()

    # Predict Test data
    chromosoms = evolution[-1]

    predictions = []
    for chromi in chromosoms:
        threshold = chromi[0]
        chrom = chromi[1:]

        predict = np.sum(data_test * chrom, 1) >= threshold
        predictions.append((predict + 1).tolist())

    np.savetxt('results/evolution_pooling/prediction_%s_%s.txt' % (pool_info, file_id), predictions, fmt='%d', delimiter=',')

    print('Done %s.' % file_id)


for pool_info in ['10_mean', '3_mean', '10_max', '3_max']:
    for use_group in [0, 1]:
        run_evolution(
            iterations, family_size, use_big_data, train_split,
            use_binary, use_zero_threshold, drop_variable,
            decomp_mode, pool_info, n_components)
