# =============================================================================
# This module compares the results derived from six algorithms:
# Total time - sum of optimising and predicting time
# Error rate
# MNLL - mean negative log likelihood
# ECE - expected calibration error
# =============================================================================
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

import scipy.io as scio

database = scio.loadmat("../datasets/benchmarks.mat")
benchmarks = database["benchmarks"][0]

dataset_names = []

for benchmark in benchmarks:
    name = benchmark[0]
    if name != "image" and name != "splice":  # didnt evaluate these two datasets
        dataset_names.append(name)

methods1 = ['db_fixed', 'db_searched', 'db_variational', 'vi', 'la', 'ep']
methods = ['GPD-variational', 'GPD-fixed', 'GPD-searched', 'LA', 'EP', 'VI']
methods_db = ['GPD-variational', 'GPD-fixed', 'GPD-searched']
result_dir = os.path.join('evaluation')
print(os.path.abspath(result_dir))


all_report = {}
all_errors = {}

# Read reports and errors
for benchmark_name in dataset_names:
    if benchmark_name not in all_report:
        all_report[benchmark_name] = {}
    if benchmark_name not in all_errors:
        all_errors[benchmark_name] = {}

    for method_name in methods1:
        summary_dir = os.path.join(result_dir, benchmark_name)
        summary_report_path = os.path.join(summary_dir, f'{method_name}_summary_report.dat')
        summary_error_path = os.path.join(summary_dir, f'{method_name}_summary_errors.dat')

        if os.path.exists(summary_report_path):
            with open(summary_report_path, 'rb') as f:
                all_report[benchmark_name][method_name] = pickle.load(f)
        else:
            print(f"Warning: {summary_report_path} does not exist.")

        if os.path.exists(summary_error_path):
            with open(summary_error_path, 'rb') as f:
                all_errors[benchmark_name][method_name] = pickle.load(f)
        else:
            print(f"Warning: {summary_error_path} does not exist.")

print("Loading process completed.")

# Initialise lists to store results
db_fixed_a_eps = []
db_searched_a_eps = []
db_variational_a_eps = []

db_fixed_error_rate = []
db_searched_error_rate = []
db_variational_error_rate = []
vi_error_rate = []
la_error_rate = []
ep_error_rate = []

db_fixed_mnll = []
db_searched_mnll = []
db_variational_mnll = []
vi_mnll = []
la_mnll = []
ep_mnll = []

db_fixed_ece = []
db_searched_ece = []
db_variational_ece = []
vi_ece = []
la_ece = []
ep_ece = []

db_fixed_time = []
db_searched_time = []
db_variational_time = []
vi_time = []

# Extract average results from summarised reports
for benchmark, methods1 in all_report.items():
    db_fixed_a_eps.append(methods1.get('db_fixed', {}).get('db_a_eps'))
    db_searched_a_eps.append(methods1.get('db_searched', {}).get('db_a_eps'))
    db_variational_a_eps.append(methods1.get('db_variational', {}).get('db_a_eps'))
    
    db_fixed_error_rate.append(methods1.get('db_fixed', {}).get('db_error_rate'))
    db_searched_error_rate.append(methods1.get('db_searched', {}).get('db_error_rate'))
    db_variational_error_rate.append(methods1.get('db_variational', {}).get('db_error_rate'))
    vi_error_rate.append(methods1.get('vi', {}).get('vi_error_rate'))
    la_error_rate.append(methods1.get('la', {}).get('la_error_rate'))
    ep_error_rate.append(methods1.get('ep', {}).get('ep_error_rate'))

    db_fixed_mnll.append(methods1.get('db_fixed', {}).get('db_mnll'))
    db_searched_mnll.append(methods1.get('db_searched', {}).get('db_mnll'))
    db_variational_mnll.append(methods1.get('db_variational', {}).get('db_mnll'))    
    vi_mnll.append(methods1.get('vi', {}).get('vi_mnll'))
    la_mnll.append(methods1.get('la', {}).get('la_mnll'))
    ep_mnll.append(methods1.get('ep', {}).get('ep_mnll'))

    db_fixed_ece.append(methods1.get('db_fixed', {}).get('db_ece'))
    db_searched_ece.append(methods1.get('db_searched', {}).get('db_ece'))
    db_variational_ece.append(methods1.get('db_variational', {}).get('db_ece'))
    vi_ece.append(methods1.get('vi', {}).get('vi_ece'))
    la_ece.append(methods1.get('la', {}).get('la_ece'))
    ep_ece.append(methods1.get('ep', {}).get('ep_ece'))

    db_fixed_time.append(
        methods1.get('db_fixed', {}).get('db_elapsed_optim')
        + methods1.get('db_fixed', {}).get('db_elapsed_pred'))
    db_searched_time.append(
        methods1.get('db_searched', {}).get('db_elapsed_optim')
        + methods1.get('db_searched', {}).get('db_elapsed_pred'))
    db_variational_time.append(
        methods1.get('db_variational', {}).get('db_elapsed_optim')
        + methods1.get('db_variational', {}).get('db_elapsed_pred'))

# Initialise lists to store errors
db_fixed_error_rate_err = []
db_searched_error_rate_err = []
db_variational_error_rate_err = []
vi_error_rate_err = []
la_error_rate_err = []
ep_error_rate_err = []

db_fixed_mnll_err = []
db_searched_mnll_err = []
db_variational_mnll_err = []
vi_mnll_err = []
la_mnll_err = []
ep_mnll_err = []

db_fixed_ece_err = []
db_searched_ece_err = []
db_variational_ece_err = []
vi_ece_err = []
la_ece_err = []
ep_ece_err = []

# Extract average errors from summarised error reports
for benchmark, methods1 in all_errors.items():
    db_fixed_error_rate_err.append(methods1.get('db_fixed', {}).get('db_error_rate'))
    db_searched_error_rate_err.append(methods1.get('db_searched', {}).get('db_error_rate'))
    db_variational_error_rate_err.append(methods1.get('db_variational', {}).get('db_error_rate'))
    vi_error_rate_err.append(methods1.get('vi', {}).get('vi_error_rate'))
    la_error_rate_err.append(methods1.get('la', {}).get('la_error_rate'))
    ep_error_rate_err.append(methods1.get('ep', {}).get('ep_error_rate'))

    db_fixed_mnll_err.append(methods1.get('db_fixed', {}).get('db_mnll'))
    db_searched_mnll_err.append(methods1.get('db_searched', {}).get('db_mnll'))
    db_variational_mnll_err.append(methods1.get('db_variational', {}).get('db_mnll'))
    vi_mnll_err.append(methods1.get('vi', {}).get('vi_mnll'))
    la_mnll_err.append(methods1.get('la', {}).get('la_mnll'))
    ep_mnll_err.append(methods1.get('ep', {}).get('ep_mnll'))
    #
    db_fixed_ece_err.append(methods1.get('db_fixed', {}).get('db_ece'))
    db_searched_ece_err.append(methods1.get('db_searched', {}).get('db_ece'))
    db_variational_ece_err.append(methods1.get('db_variational', {}).get('db_ece'))
    vi_ece_err.append(methods1.get('vi', {}).get('vi_ece'))
    la_ece_err.append(methods1.get('la', {}).get('la_ece'))
    ep_ece_err.append(methods1.get('ep', {}).get('ep_ece'))

# Set path to save figures
graph_dir = os.path.join('graph')
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)

measurements = ["Error rate", "MNLL", "ECE"]
times = ["Time"]

# Aggregate average results to dictionary data
data = {
    "Error rate": {
        "GPD-fixed": db_fixed_error_rate,
        "GPD-searched": db_searched_error_rate,
        "GPD-variational": db_variational_error_rate,
        "VI": vi_error_rate,
        "LA": la_error_rate,
        "EP": ep_error_rate
    },
    "MNLL": {
        "GPD-fixed": db_fixed_mnll,
        "GPD-searched": db_searched_mnll,
        "GPD-variational": db_variational_mnll,
        "VI": vi_mnll,
        "LA": la_mnll,
        "EP": ep_mnll
    },
    "ECE": {
        "GPD-fixed": db_fixed_ece,
        "GPD-searched": db_searched_ece,
        "GPD-variational": db_variational_ece,
        "VI": vi_ece,
        "LA": la_ece,
        "EP": ep_ece
    }
}

# Aggregate average errors to disctionary errors
errors = {
    "Error rate": {
        "GPD-fixed": db_fixed_error_rate_err,
        "GPD-searched": db_searched_error_rate_err,
        "GPD-variational": db_variational_error_rate_err,
        "VI": vi_error_rate_err,
        "LA": la_error_rate_err,
        "EP": ep_error_rate_err
    },
    "MNLL": {
        "GPD-fixed": db_fixed_mnll_err,
        "GPD-searched": db_searched_mnll_err,
        "GPD-variational": db_variational_mnll_err,
        "VI": vi_mnll_err,
        "LA": la_mnll_err,
        "EP": ep_mnll_err
    },
    "ECE": {
        "GPD-fixed": db_fixed_ece_err,
        "GPD-searched": db_searched_ece_err,
        "GPD-variational": db_variational_ece_err,
        "VI": vi_ece_err,
        "LA": la_ece_err,
        "EP": ep_ece_err
    }
}

# Aggregate time to dictionary time_db: only compare time within GPD algorithms
time_db = {
    "Time": {
        "GPD-fixed": db_fixed_time,
        "GPD-searched": db_searched_time,
        "GPD-variational": db_variational_time}
    }

# Aggregate a_eps to dictionary e_eps_values: to show optimal a_eps for each GPD algorithm
a_eps_values = {
    'GPD-fixed': db_fixed_a_eps,
    'GPD-searched': db_searched_a_eps,
    'GPD-variational': db_variational_a_eps
}

# Rank the six algorithms based on measurement values - 1 represents best and 6 represents worst
rank_data = {metric: {} for metric in data}

for metric, metric_data in data.items():
    for method in methods:
        rank_data[metric][method] = []

    # Compute rank for each dataset
    for i in range(len(dataset_names)):
        values = [metric_data[method][i] for method in methods]
        ranks = stats.rankdata(values)  

        for j, method in enumerate(methods):
            rank_data[metric][method].append(ranks[j])

# Compute mean and std dev for rank
avg_rank = {metric: {} for metric in data}
std_rank = {metric: {} for metric in data}

for metric in data:
    for method in methods:
        avg_rank[metric][method] = np.mean(rank_data[metric][method])
        std_rank[metric][method] = np.std(rank_data[metric][method])

### =======================================================================================
# Plot results
# Colors and markers for the plots
num_datasets = len(dataset_names)
num_cols = 3
num_rows = (num_datasets + num_cols - 1) // num_cols

# Plot measurement values - Error rate, MNLL, ECE
for measurement in measurements:
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 18))
    axs = axs.flatten()

    for i, dataset in enumerate(dataset_names):
        ax = axs[i]

        values = [data[measurement][method][i] for method in methods]
        errs = [errors[measurement][method][i] for method in methods]

        x_pos = range(len(methods))

        for j, method in enumerate(methods):
            ax.errorbar(x_pos[j], values[j], yerr=errs[j], fmt='o',
                        ecolor='black', capsize=4, markersize=8, linestyle='None')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.set_ylabel('Value')
        ax.set_title(f'{dataset}')

    # Add rank plot at the last one
    if len(dataset_names) < num_cols * num_rows:
        ax_rank = axs[len(dataset_names)] 
        avg_values = [avg_rank[measurement][method] for method in methods]
        std_values = [std_rank[measurement][method] for method in methods]

        for j, method in enumerate(methods):
            ax_rank.errorbar(j, avg_values[j], yerr=std_values[j], fmt='o',
                             ecolor='black', capsize=4, markersize=8, linestyle='None', 
                             color=ax.errorbar(x_pos[j], values[j], yerr=errs[j], fmt='o', 
                             markersize=8, linestyle='None').lines[0].get_color())  # set same colors as previous scatter plots

        ax_rank.set_xticks(range(len(methods)))
        ax_rank.set_xticklabels(methods, rotation=45, ha='right')
        ax_rank.set_title(f'Average Rank', fontsize=12)
        ax_rank.set_ylabel('Rank (lower is better)', fontsize=12)

    fig.suptitle(f'{measurement} Comparison', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(graph_dir, f'{measurement}_comparison.pdf')
    plt.savefig(save_path, format="pdf")

    plt.show()

# Draw independent rank plots
rank_fig, rank_axs = plt.subplots(1, len(measurements), figsize=(18, 6)) 

for m_idx, measurement in enumerate(measurements):
    avg_values = [avg_rank[measurement][method] for method in methods]
    std_values = [std_rank[measurement][method] for method in methods]

    for j, method in enumerate(methods):
        color = axs[0].errorbar(j, values[j], yerr=errs[j], fmt='o',
                                markersize=8, linestyle='None').lines[0].get_color()
        rank_axs[m_idx].errorbar(j, avg_values[j], yerr=std_values[j], fmt='o',
                                  ecolor='black', capsize=4, markersize=8, linestyle='None', 
                                  color=color)

    rank_axs[m_idx].set_xticks(range(len(methods)))
    rank_axs[m_idx].set_xticklabels(methods, rotation=45, ha='right')
    rank_axs[m_idx].set_title(f'Average Rank for {measurement}', fontsize=12)
    rank_axs[m_idx].set_ylabel('Rank (lower is better)', fontsize=12)

rank_fig.suptitle('Rank Comparison', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])
rank_save_path = os.path.join(graph_dir, 'rank_comparison.pdf')
plt.savefig(rank_save_path, format="pdf")

plt.show()

# Draw Time comparison
colors = ['lightcoral', 'lightblue', 'lightgreen']
for time in times:
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 18))
    axs = axs.flatten()

    legend_bars = [] 

    for i, dataset in enumerate(dataset_names):
        ax = axs[i]

        values = [time_db[time][method][i] for method in methods_db]
        # Get a_eps value for each method
        a_eps_values_for_methods = [a_eps_values[method][i] for method in methods_db]

        # Adjust labels to ensure the bars not overlapping
        unique_labels = []
        for j, a_eps in enumerate(a_eps_values_for_methods):
            adjusted_label = f'{a_eps:.3f}'
            while adjusted_label in unique_labels:
                a_eps += 1e-6 
                adjusted_label = f'{a_eps:.3f}'
            unique_labels.append(adjusted_label)
            
        bars = ax.bar(unique_labels, values, color=colors[:len(methods_db)])

        for bar, method in zip(bars, methods_db):
            bar.set_label(method)
            legend_bars.append(bar)

        ax.set_xlabel('a_eps Values')
        ax.set_yscale("log")  # set yscale as log
        ax.set_ylabel('Time')  
        ax.set_title(f'{dataset}')

    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(f'{time.replace("_", " ").title()} Comparison', fontsize=20)
    
    fig.legend(legend_bars, methods_db, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=len(methods_db), fontsize=12)

    plt.tight_layout(rect=[0, 0.15, 1, 0.96])  

    save_path = os.path.join(graph_dir, f'time comparison.pdf')
    plt.savefig(save_path, format="pdf")
    plt.show()

