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
from datetime import datetime
import logging
import scipy

EXPERIMENTS_DIR = '.'
DATASETS_FILE = 'datasets.h5'


logging.basicConfig(
    filename='example.log',
    format='%(asctime)s %(levelname)-8s %(message)s',
    filemode='w',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True)


#def yield_parameters_inner_loop(parameters: list, number_of_parameters: int):
#    if number_of_parameters == 0:
#        yield []#
#
#    for index in range(len(parameters) - number_of_parameters + 1):
#        yield parameters[index] + yield_parameters_inner_loop[parameters]


def get_parameter_axis(df: pd.DataFrame):
    """
    Returns the parameter axis from the dataframe.

    Because the dataframes can contain a variable number of
    parameters, this command removes the ID and class columns
    and returns the remainder.
    """
    parameter_axis = list(df.columns)
    parameter_axis.remove('ID')
    parameter_axis.remove('class')
    return parameter_axis


def evaluate_classification_performance(df, axes, number_of_iterations=100, fraction_of_data_to_use=1.0):
    classifier_data = df[axes]
    classifier_labels = df['class']
    classification_performances = list()

    for i in range(number_of_iterations):
        svm_data = np.array(classifier_data)
        svm_labels = np.array(classifier_labels)

        indices = np.arange(svm_data.shape[0])
        np.random.shuffle(indices)

        svm_data = svm_data[indices]
        svm_labels = svm_labels[indices]

        split_index = int(len(svm_data) / 2. * fraction_of_data_to_use)
        end_index = int(len(svm_data) * fraction_of_data_to_use)
        train_data = svm_data[:split_index]
        train_labels = svm_labels[:split_index]
        test_data = svm_data[split_index:end_index]
        test_labels = svm_labels[split_index:end_index]

        clf = svm.SVC()

        clf.fit(train_data, train_labels)
        classification_performances.append(np.sum(clf.predict(test_data) == test_labels) / len(test_data))

    standard_deviation = np.std(classification_performances, ddof=1) if len(classification_performances) > 1 else None

    return np.mean(classification_performances), standard_deviation, classification_performances


def generate_df(key_name, fn):
    num_samples = 1000
    logging.info("Generating new dataset!")
    df = generate_pituitary_dataset(parameter_function=fn,
                                    num_samples=num_samples,
                                    classify=True,
                                    retain_trajectories=False)

    logging.info(f"Storing generated dataset with {num_samples} samples, using key '{key_name}'")
    with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
        store[key_name] = df
        logging.info("New dataset stored!")
    return df


def experiment(generate_new_dataset: bool, rerun_experiments: bool, parameter_function, slug):
    if os.path.isfile(DATASETS_FILE):
        logging.info(f"H5 file {DATASETS_FILE} exists")
    else:
        logging.info(f"H5 file {DATASETS_FILE} not found")

    experiment_h5_key = '/experiment_' + slug

    df = generate_or_load_dataset(parameter_function, generate_new_dataset, experiment_h5_key)

    with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE)) as store:
        experiment_dataset_exists = experiment_h5_key + '/dataset' in list(store.keys())
        if experiment_dataset_exists:
            logging.info(f"Found dataset '{experiment_h5_key}' in the HDF5 data store")
        else:
            logging.info(f"Dataset not present in HDF5 store (checked for '{experiment_h5_key}')")

    should_rerun_experiments = generate_new_dataset or rerun_experiments or not experiment_dataset_exists

    parameter_axis = get_parameter_axis(df)
    # The HDF5 file is CLOSED here so subsequent functions can use it
    find_best_parameter_combinations(df, parameter_axis, experiment_h5_key, should_rerun_experiments)


def find_best_parameter_combinations(df, parameter_axis, experiment_h5_key, should_rerun_experiments, pvalue_threshold=0.01):
    for dimension in range(1, len(parameter_axis)):
        output_key = experiment_h5_key + f'/outputs/dim_{dimension}'
        experiment_is_running_key = output_key + 'currently_running'

        if should_rerun_experiments:
            # Force-run the experiment
            logging.info(f"Forcing new experiment, dimension {dimension}")
            should_run_experiment = True
            experiment_is_running = False
            # Note - ensure that no two jobs force-run the same experiment! This could lead to a collision!
        else:
            # Check if the experiment has been run or not
            with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE), 'a') as store:
                if output_key in store.keys():
                    should_run_experiment = False
                else:
                    should_run_experiment = True

                if experiment_is_running_key in store.keys():
                    experiment_is_running = True
                else:
                    experiment_is_running = False

        if not should_run_experiment:
            logging.info(f"Experiment already run. Skipping dimension {dimension}")
            continue
        elif experiment_is_running:
            logging.info(f"Experiment for dimension {dimension} should be run, but another job has already claimed it. Skipping")
            continue

        logging.info(f"Running experiments for dimension {dimension}")
        approximate_classification_performances = dict()
        logging.info(
            f"There are {len(list(itertools.combinations(parameter_axis, dimension)))} combinations to evaluate!")
        for iteration_index, parameters in enumerate(itertools.combinations(parameter_axis, dimension)):
            if iteration_index % 1000 == 0:
                axis_string = ', '.join(parameters)
                logging.info(f"Evaluating axis {iteration_index}: '{axis_string}'")
            mean_performance, standard_deviation, performances = evaluate_classification_performance(df, list(parameters), fraction_of_data_to_use=0.2)
            approximate_classification_performances[parameters] = (mean_performance, standard_deviation, performances)

        ordered_approximate_classification_performances = OrderedDict(sorted(approximate_classification_performances.items(), key=lambda x: x[1][0], reverse=True))

        # Here we create our list of the 'top' classification performances,
        # i.e. those that we cannot distinguish without further analysis
        indistinguishable_approximate_classification_performances = list()
        top_key, top_entry = list(ordered_approximate_classification_performances.items())[0]
        for key, (mean_performance, standard_deviation, performances) in ordered_approximate_classification_performances.items():
            tvalue, pvalue = scipy.stats.ttest_ind(top_entry[2], performances, nan_policy='omit')

            if pvalue < pvalue_threshold:
                break
            indistinguishable_approximate_classification_performances.append(key)

        logging.info(f"{len(indistinguishable_approximate_classification_performances)} parameter sets are indistinguishable. Refining by using all data points")

        classification_performances = dict()
        for parameters in indistinguishable_approximate_classification_performances:
            mean_performance, standard_deviation, performances = evaluate_classification_performance(df,
                                                                                                     list(parameters))
            classification_performances[parameters] = (mean_performance, standard_deviation, performances)

        ordered_classification_performances = OrderedDict(sorted(classification_performances.items(), key=lambda x: x[1][0], reverse=True))

        # Here we create our list of the 'top' classification performances,
        # i.e. those that we cannot distinguish without further analysis
        indistinguishable_classification_performances = list()
        top_key, top_entry = list(ordered_classification_performances.items())[0]
        for key, (mean_performance, standard_deviation, performances) in ordered_classification_performances.items():
            tvalue, pvalue = scipy.stats.ttest_ind(top_entry[2], performances, nan_policy='omit')

            if pvalue < pvalue_threshold:
                continue

            item = [int(item in key) for item in parameter_axis]
            item += [mean_performance, standard_deviation, tvalue, pvalue]
            indistinguishable_classification_performances.append(item)

        logging.info(f"{len(indistinguishable_classification_performances)} parameter sets are still indistinguishable.")

        results_dataframe = pd.DataFrame(data=indistinguishable_classification_performances,
                                         columns=parameter_axis + ['mean', 'stddev', 'tvalue', 'pvalue'])

        with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE), 'a') as store:
            data_key = output_key + '/data'
            store[data_key] = results_dataframe


def find_best_parameter_combinations_old(df, parameter_axis):
    approximate_classification_performance_for_all_dimensions = list()
    for axis_index in range(1, len(parameter_axis)):
        logging.info(f"Running experiments for dimension {axis_index}")
        approximate_classification_performances = dict()
        logging.info(
            f"There are {len(list(itertools.combinations(parameter_axis, axis_index)))} combinations to evaluate!")
        for iteration_index, parameters in enumerate(itertools.combinations(parameter_axis, axis_index)):
            if iteration_index % 1000 == 0:
                axis_string = ', '.join(parameters)
                logging.info(f"Evaluating axis {iteration_index}: '{axis_string}'")
            mean_performance, standard_deviation, performances = evaluate_classification_performance(df, list(parameters))
            approximate_classification_performances[parameters] = (mean_performance, standard_deviation, performances)

        ordered_approximate_classification_performances = OrderedDict(sorted(approximate_classification_performances.items(), key=lambda x: x[1][0], reverse=True))
        approximate_classification_performance_for_all_dimensions.append(ordered_approximate_classification_performances)
    dimensions_out = list()
    for approximate_classification_performances in approximate_classification_performance_for_all_dimensions:
        outputs = list()

        # Here we create our list of the 'top' classification performances,
        # i.e. those that we cannot distinguish without further analysis
        indistinguishable_classification_performances = list()
        top_key, top_entry = list(approximate_classification_performances.items())[0]
        for key, (mean_performance, standard_deviation, performances) in approximate_classification_performances.items():
            tvalue, pvalue = scipy.stats.ttest_ind(top_entry[2], performances, nan_policy='omit')

            if pvalue < 0.01:
                break
            indistinguishable_classification_performances.append(key)
            output = (key, mean_performance, standard_deviation, tvalue, pvalue)
            outputs.append(output)
        dimensions_out.append(outputs)


def generate_or_load_dataset(parameter_function, generate_new_dataset: bool, experiment_key: str):
    """
    Loads a dataset if it exists, generates it if it doesn't exist or the user requests it
    """
    dataset_key = experiment_key + '/dataset'
    if generate_new_dataset:
        logging.info("New dataset requested. Forcing generation")
        df = generate_df(dataset_key, parameter_function)
    else:
        logging.info("Checking if dataset exists in H5 file")
        with pd.HDFStore(os.path.join(EXPERIMENTS_DIR, DATASETS_FILE), 'a') as store:
            if dataset_key in list(store.keys()):
                logging.info("Loading existing dataset")
                df = store[dataset_key]
                logging.info(f"Existing dataset contains {len(df)} items")
            else:
                logging.info("Dataset does not exist. Closing the h5py file")
                store.close()  # We no longer need the store and we should close it
                df = generate_df(dataset_key, parameter_function)
    return df


def experiment_basic(generate_new_dataset: bool = False):
    logging.info("Running experiment_basic")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters, 'basic')


def experiment_Isk(generate_new_dataset: bool = False):
    logging.info("Running experiment_Isk")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk, 'Isk')


def experiment_Isk_Ibk(generate_new_dataset: bool = False):
    logging.info("Running experiment_Isk_Ibk")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk, 'Isk_Ibk')


def experiment_Isk_Ibk_Ikir(generate_new_dataset: bool = False):
    logging.info("Running experiment_Isk_Ibk_Ikir")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk_Ikir, 'Isk_Ibk_Ikir')


def experiment_Isk_Ibk_Ikir_Icat(generate_new_dataset: bool = False):
    logging.info("Running experiment_Isk_Ibk_Ikir_Icat")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat, 'Isk_Ibk_Ikir_Icat')


def experiment_Isk_Ibk_Ikir_Icat_Ia(generate_new_dataset: bool = False):
    logging.info("Running experiment_Isk_Ibk_Ikir_Icat_Ia")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia, 'Isk_Ibk_Ikir_Icat_Ia')


def experiment_Isk_Ibk_Ikir_Icat_Ia_Inav(generate_new_dataset: bool = False):
    logging.info("Running experiment_Isk_Ibk_Ikir_Icat_Ia_Inav")
    experiment(generate_new_dataset, False, pituitary_ori_ode_parameters_Isk_Ibk_Ikir_Icat_Ia_Inav,
               'Isk_Ibk_Ikir_Icat_Ia_Inav')


if __name__ == '__main__':
    #experiment_basic(True)
    experiment_Isk(True)
    #experiment_Isk_Ibk(False)
    #experiment_Isk_Ibk_Ikir(False)
    #experiment_Isk_Ibk_Ikir_Icat(False)
    #experiment_Isk_Ibk_Ikir_Icat_Ia(False)
    #experiment_Isk_Ibk_Ikir_Icat_Ia_Inav(False)
