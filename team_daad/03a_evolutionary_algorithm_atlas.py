import numpy as np
import pandas as pd
from scipy.stats import zscore
from itertools import permutations
from pylab import *

# Paramters
iterations = 250
family_size = 512
use_big_data = False
train_split = 0.2
use_binary = False
use_zero_threshold = True
drop_variable = 'Scanner'
use_group = 0


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
                  use_binary, use_zero_threshold, drop_variable):

    # Import metadata to know who is control and who is patient
    df = pd.read_csv('data/PAC2018_Covariates_detailed.csv', index_col=0)

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

    # Drop outliers
    data = df.drop(['Label'], 1)
    header = data.keys()
    outliers = np.sum((np.abs(zscore(data)) > 5), 1) != 0
    data = np.array(data)[np.invert(outliers)]
    labels = labels[np.invert(outliers)]
    sub = sub[np.invert(outliers)]

    # Specify chromosoms
    chrom_size = data.shape[1]

    # Create linear combination with 'Age', 'Gender', 'TIV', 'Scanner' & 'Tvoxels'

    aegaerg
    if use_big_data:
        for i in range(int(np.where(header == 'Tvoxels')[0])):
            data = np.hstack(
                (data, data[:, i:chrom_size] * data[:, i][..., None]))
            header = np.hstack(
                (header, ['_'.join([header[i], h]) for h in header[i:chrom_size]]))
        chrom_size = data.shape[1]

    # zscore data to prepare for machine learning
    data = zscore(data)

    # Create new family
    family = [get_new_chromosom(chrom_size, binary=use_binary,
                                zero_threshold=use_zero_threshold)
              for x in range(family_size)]

    result_train = []
    result_test = []
    result_thresh = []
    evolution = []

    for k in range(iterations):

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
        sub_balanced = sub[new_data_id]

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

        good_parents = fit_train.argsort()[-16:]

        # Save best chromosom
        evolution.append([family[good_parents[-1]],
                          family[good_parents[-2]],
                          family[good_parents[-3]]])

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
            for j in [[0, 25], [25, 50], [50, 75], [75, 100]]:
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

    title_text = ' Acc = %s - Iter: %04d - Family: %04d - Big: %s - ' % (
        round(acc_train, 2), iterations, family_size, use_big_data)
    title_text += 'Binary: %s - ZeroThresh: %s - Group: %s' % (
        use_binary, use_zero_threshold, postfix)
    file_id = 'iter_%04d_family_%04d_split_%02d_big_%s_bin_%s_zeroThresh_%s_group_%s' % (
        iterations, family_size, (100 * train_split),
        use_big_data, use_binary, use_zero_threshold, postfix)
    if drop_variable != '':
        title_text += ' - Dropped: %s' % drop_variable
        file_id += '_dropped_%s' % drop_variable

    figure(figsize=(16, 6))
    plot(result_train)
    plot(result_test)
    plot(np.array(result_thresh) + 60)
    legend(['Train [~%0.1f]' % np.mean(result_train[250:]),
            'Test [~%0.1f]' % np.mean(result_test[250:]),
            'Threshold [+60]'])
    title('Fitness:%s - Threshold at %f' % (title_text, result_thresh[-1]))
    xlabel('Generation')
    ylabel('Accuracy [%]')
    tight_layout()
    savefig('results/evolution/fitness_%s.png' % (file_id))
    close()

    evolution = np.array([np.array([[float(f[0])] + list(f[1].astype('float'))
                                    for f in ev]) for ev in evolution])
    evolutionRGB = np.rollaxis(np.rollaxis(evolution, 2), 1).astype('float32')
    figure(figsize=(16, 8))
    imshow(evolutionRGB, aspect='auto')
    title('Chromosom:%s - Threshold at %f' % (title_text, result_thresh[-1]))
    ylabel('Generation')
    xticks(range(chrom_size + 1), ['Threshold'] + header.tolist(),
           rotation='vertical')
    subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.15)
    savefig('results/evolution/chromosom_%s.png' % (file_id))
    close()

    family = np.array([[float(f[0])] + list(f[1].astype('float'))
                       for f in family])
    figure(figsize=(16, 8))
    imshow(family, aspect='auto')
    title('Final Family:%s - Threshold at %f' %
          (title_text, result_thresh[-1]))
    ylabel('Generation')
    xticks(range(chrom_size + 1), ['Threshold'] + header.tolist(),
           rotation='vertical')
    subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.15)
    savefig('results/evolution/family_%s.png' % (file_id))
    close()


for use_big_data in [False, True]:
    for use_zero_threshold in [True, False]:
        for use_binary in [True, False]:

                run_evolution(iterations, family_size, use_big_data, train_split,
                              use_binary, use_zero_threshold, drop_variable)
