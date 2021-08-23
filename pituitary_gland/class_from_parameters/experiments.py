import itertools

import pandas as pd
from traja.dataset.pituitary_gland import pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia_Inav, \
    generate_pituitary_dataset, pituitary_ori_ode_parameters_Isk, pituitary_ori_ode_parameters, \
    pituitary_ori_ode_parameters_Isk_Ibk, pituitary_ori_ode_parameters_Isk_Ibk_Ikir, \
    pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia
import os
import numpy as np
from sklearn import svm
from collections import OrderedDict
from sklearn import neural_network
import scipy

EXPERIMENTS_DIR = '.'
DATASETS_FILE = 'datasets.h5'


def yield_parameters_inner_loop(parameters: list, number_of_parameters: int):
    if number_of_parameters == 0:
        yield []

    for index in range(len(parameters) - number_of_parameters + 1):
        yield parameters[index] + yield_parameters_inner_loop[parameters]


def get_parameter_axis(df: pd.DataFrame):
    parameter_axis = list(df.columns)
    parameter_axis.remove('ID')
    parameter_axis.remove('class')
    return parameter_axis


def test_classification_performance(df, axes):
    classifier_data = df[axes]
    classifier_labels = df['class']
    classification_performances = list()

    iter_count = 100

    for i in range(iter_count):
        svm_data = np.array(classifier_data)
        svm_labels = np.array(classifier_labels)

        indices = np.arange(svm_data.shape[0])
        np.random.shuffle(indices)

        svm_data = svm_data[indices]
        svm_labels = svm_labels[indices]

        split_index = int(len(svm_data) / 2.)
        train_data = svm_data[:split_index]
        train_labels = svm_labels[:split_index]
        test_data = svm_data[split_index:]
        test_labels = svm_labels[split_index:]

        clf = svm.SVC()

        clf.fit(train_data, train_labels)
        classification_performances.append(np.sum(clf.predict(test_data) == test_labels) / len(test_data))

    standard_deviation = np.std(classification_performances, ddof=1) if len(classification_performances) > 1 else None

    return np.mean(classification_performances), standard_deviation, classification_performances


def experiment(generate_new_dataset: bool, fn, slug):
    if not os.path.isfile(DATASETS_FILE) or generate_new_dataset:
        print("Generating new dataset!")
        df = generate_pituitary_dataset(parameter_function=fn,
                                        num_samples=100000,
                                        classify=True,
                                        retain_trajectories=False)

        with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
            store['dataset_' + slug] = df
    else:
        print("Loading existing dataset!")
        with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
            df = store['dataset_' + slug]
    print(df)

    parameter_axis = get_parameter_axis(df)
    values_list = list()
    for index in range(1, len(parameter_axis)):
        print("Running experiments for dimension", index)
        values = dict()
        for item in itertools.combinations(parameter_axis, index):
            mean_performance, standard_deviation, performances = test_classification_performance(df, list(item))
            values[item] = (mean_performance, standard_deviation, performances)

        values_ordered = OrderedDict(sorted(values.items(), key=lambda x: x[1][0], reverse=True))
        values_list.append(values_ordered)

    dimensions_out = list()
    for dimension in values_list:
        outputs = list()
        top_entry = list(dimension.values())[0]
        for key, value in dimension.items():
            tvalue, pvalue = scipy.stats.ttest_ind(top_entry[2], value[2], nan_policy='omit')
            output = (key, value[0], value[1], tvalue, pvalue)
            outputs.append(output)
        dimensions_out.append(outputs)
    with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
        store['outputs_' + slug] = df


def experiment_basic(generate_new_dataset: bool = False):
    print("Running experiment_basic")
    experiment(generate_new_dataset, pituitary_ori_ode_parameters, '')


def experiment_Isk(generate_new_dataset: bool = False):
    print("Running experiment_Isk")
    experiment(generate_new_dataset, pituitary_ori_ode_parameters_Isk, 'Isk')


def experiment_Isk_Ibk(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk")
    experiment(generate_new_dataset, pituitary_ori_ode_parameters_Isk_Ibk, 'Isk_Ibk')


def experiment_Isk_Ibk_Ikir(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk_Ikir")
    experiment(generate_new_dataset, pituitary_ori_ode_parameters_Isk_Ibk_Ikir, 'Isk_Ibk_Ikir')


def experiment_Isk_Ibk_Ikir_Icat(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk_Ikir_Icat")
    experiment(generate_new_dataset, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat, 'Isk_Ibk_Ikir_Icat')


def experiment_Isk_Ibk_Ikir_Icat_Ia(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk_Ikir_Icat_Ia")
    experiment(generate_new_dataset, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia, 'Isk_Ibk_Ikir_Icat_Ia')


def experiment_Isk_Ibk_Ikir_Icat_Ia_Inav(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk_Ikir_Icat_Ia_Inav")
    experiment(generate_new_dataset, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia_Inav, 'Isk_Ibk_Ikir_Icat_Ia_Inav')


experiment_basic(False)
experiment_Isk(False)
experiment_Isk_Ibk(False)
experiment_Isk_Ibk_Ikir(False)
experiment_Isk_Ibk_Ikir_Icat(False)
experiment_Isk_Ibk_Ikir_Icat_Ia(False)
experiment_Isk_Ibk_Ikir_Icat_Ia_Inav(False)
