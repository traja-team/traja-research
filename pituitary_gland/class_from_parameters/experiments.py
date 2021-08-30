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


def generate_df(key_name, fn):
    print("Generating new dataset!")
    df = generate_pituitary_dataset(parameter_function=fn,
                                    num_samples=100,
                                    classify=True,
                                    retain_trajectories=False)

    with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
        store[key_name] = df
        print("New dataset stored!")
    return df


def experiment(generate_new_dataset: bool, rerun_experiments: bool, fn, slug):
    dataset_key = 'dataset_' + slug
    if generate_new_dataset:
        df = generate_df(dataset_key, fn)
    else:
        if os.path.isfile(DATASETS_FILE):
            print("Checking if dataset exists in H5 file!!")
            with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
                if '/' + dataset_key in list(store.keys()):
                    print("Loading existing dataset!")
                    df = store[dataset_key]
                else:
                    df = generate_df(dataset_key, fn)
        else:
            df = generate_df(dataset_key, fn)

    print(df)

    outputs_key = 'outputs_' + slug

    with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
        outputs_exists = '/' + outputs_key in list(store.keys())

    if generate_new_dataset or rerun_experiments or not outputs_exists:
        print("Running experiment")

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
            store[outputs_key] = df
    else:
        print("Experiment already run!")


def experiment_basic(generate_new_dataset: bool = False):
    print("Running experiment_basic")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters, '')


def experiment_Isk(generate_new_dataset: bool = False):
    print("Running experiment_Isk")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk, 'Isk')


def experiment_Isk_Ibk(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk, 'Isk_Ibk')


def experiment_Isk_Ibk_Ikir(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk_Ikir")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk_Ikir, 'Isk_Ibk_Ikir')


def experiment_Isk_Ibk_Ikir_Icat(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk_Ikir_Icat")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat, 'Isk_Ibk_Ikir_Icat')


def experiment_Isk_Ibk_Ikir_Icat_Ia(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk_Ikir_Icat_Ia")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia, 'Isk_Ibk_Ikir_Icat_Ia')


def experiment_Isk_Ibk_Ikir_Icat_Ia_Inav(generate_new_dataset: bool = False):
    print("Running experiment_Isk_Ibk_Ikir_Icat_Ia_Inav")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia_Inav,
               'Isk_Ibk_Ikir_Icat_Ia_Inav')


experiment_basic(False)
experiment_Isk(False)
experiment_Isk_Ibk(False)
experiment_Isk_Ibk_Ikir(False)
experiment_Isk_Ibk_Ikir_Icat(False)
experiment_Isk_Ibk_Ikir_Icat_Ia(False)
experiment_Isk_Ibk_Ikir_Icat_Ia_Inav(False)
