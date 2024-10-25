# =============================================================================
# This module evaluates six algorithms on 10 splits of each benchmark dataset
# derived from the UCI, DELVE and STATLOG repositories:
# https://github.com/tdiethe/gunnar\_raetsch\_benchmark\_datasets/
# =============================================================================

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import pickle
import numpy as np
import tensorflow as tf
import gpflow
import GPy

sys.path.insert(0, "src")
import datasets
import evaluation

import scipy.io as scio

database = scio.loadmat("./datasets/benchmarks.mat")
benchmarks = database["benchmarks"][0]

# Convert results to save in reports
def convert_to_serializable(obj):
    if isinstance(obj, gpflow.base.Parameter):
        return obj.numpy()  
    elif isinstance(obj, tf.Tensor):
        return obj.numpy()  
    elif isinstance(obj, GPy.core.parameterization.param.Param):
        return obj.values  
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj

# To save reports for each split and summarised results
def experiment(method_name, method_evaluation, benchmarks, database):
    all_reports = {}
    all_errors = {}

    prefix = method_name[:2]

    for benchmark in benchmarks:
        name = benchmark[0]
        if name == "image" or name == "splice":  # these two are multi classes
            continue

        splits = 10
        data = database[name][0, 0]

        for split_index in range(splits):
            tf.random.set_seed(split_index)
            np.random.seed(split_index)
            X, y, Xtest, ytest = datasets.get_split_data(data, name, split_index)
            X, Xtest = datasets.normalise_oneminusone(X, Xtest)

            ARD = False
            report = {}
            report["ARD"] = ARD
            report["training_size"] = X.shape[0]
            report["test_size"] = Xtest.shape[0]

            ytest = ytest.astype(int)
            report["ytest"] = ytest


            Z = None
            method_report = method_evaluation(X, y, Xtest, ytest, ARD, Z)

            if np.isnan(method_report.get(f"{prefix}_error_rate", 0)) or np.isnan(method_report.get(f"{prefix}_mnll", 0)) or np.isnan(
                    method_report.get(f"{prefix}_ece", 0)):
                print(f"Skipping split {split_index} for benchmark {name} due to NaN values in the report.")
                continue

            report.update(method_report)

            # save results of the current split
            result_dir = os.path.join('results', 'evaluation', name, f'split_{split_index}')
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            report_path = os.path.join(result_dir, f'{method_name}_report.dat')
            with open(report_path, 'wb') as f:
                pickle.dump(convert_to_serializable(report), f)

            # save summarised results
            if name not in all_reports:
                all_reports[name] = []
            all_reports[name].append(report)

        # compute average and error for benchmarks
        average_report = {}
        error_report = {}
        num_splits = len(all_reports[name])
        keys = all_reports[name][0].keys()

        for key in keys:
            if isinstance(all_reports[name][0][key], (int, float, np.number)):
                values = [report[key] for report in all_reports[name]]
                average_report[key] = np.mean(values)
                error_report[key] = np.std(values) / np.sqrt(num_splits)
            else:
                average_report[key] = all_reports[name][0][key]
                error_report[key] = None

        all_reports[name] = convert_to_serializable(average_report)
        all_errors[name] = convert_to_serializable(error_report)

        # save summarised report
        summary_dir = os.path.join('results', 'evaluation', name)
        summary_report_path = os.path.join(summary_dir, f'{method_name}_summary_report.dat')
        summary_error_path = os.path.join(summary_dir, f'{method_name}_summary_errors.dat')

        with open(summary_report_path, 'wb') as f:
            pickle.dump(all_reports[name], f)
        with open(summary_error_path, 'wb') as f:
            pickle.dump(all_errors[name], f)


# To evaluate GPD-fixed with a_eps = 0.1
def evaluate_db1(X, y, Xtest, ytest, ARD, Z):
    db_report = evaluation.evaluate_db_old(X, y, Xtest, ytest, a_eps=0.1, ARD=ARD, Z=Z)
    return db_report
experiment('gpd_fixed', evaluate_db1, benchmarks, database)

# To evaluate GPD-searched with optimal_a_eps by random search
def evaluate_db2(X, y, Xtest, ytest, ARD, Z):
    optimal_a_eps, optimization_time = evaluation.optimize_a_eps(X, y, Xtest, ytest, ARD=ARD, Z=Z)
    db_report = evaluation.evaluate_db_old(X, y, Xtest, ytest, optimal_a_eps, ARD=ARD, Z=Z)
    db_report["db_elapsed_optim"] += optimization_time
    return db_report
experiment('db_searched', evaluate_db2, benchmarks, database)

# To evaluate other four algorithms
experiment('gpd_variational', evaluation.evaluate_db_new, benchmarks, database)

experiment('vi', evaluation.evaluate_vi, benchmarks, database)

experiment('la', evaluation.evaluate_la, benchmarks, database)

experiment('ep', evaluation.evaluate_ep, benchmarks, database)

