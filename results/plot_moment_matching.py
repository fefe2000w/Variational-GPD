# =============================================================================
# This module is to visulise moment matching of lognormal and gamma distributions
# =============================================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm

graph_dir = os.path.join('graph')
if not os.path.exists(graph_dir):
    os.makedirs(graph_dir)


alphas = [0.01, 0.1, 0.8, 1.2]  # different parameters for Gamma distribution
x = np.linspace(0, 10, 1000) 
colors = ['b', 'g', 'r', 'm']  

plt.figure(figsize=(12, 6)) 

# Draw log-PDF
plt.subplot(1, 2, 1) 
for i, alpha in enumerate(alphas):
    color = colors[i]  # get corresponding color

    # Gamma distribution
    gamma_pdf = gamma.pdf(x, a=alpha, scale=1)
    gamma_logpdf = np.log(gamma_pdf)  # take logarithm

    # Log-normal distribution parameters
    sigma = np.sqrt(np.log(1 + 1/alpha))
    mu = np.log(alpha) - 0.5 * sigma**2
    lognormal_pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
    lognormal_logpdf = np.log(lognormal_pdf)  # Take logarithm

    # plot log-PDF
    plt.plot(x, gamma_logpdf, label=f'Gamma(α={alpha})', linestyle='--', color=color)
    plt.plot(x, lognormal_logpdf, label=f'Log-normal(α={alpha})', color=color)

plt.title('Log Probability Density Functions (log-PDF)')
plt.xlabel('x')
plt.ylabel('Log Density')
plt.xlim(0, 4)  # set x-axis range to [0, 4]
plt.ylim(-15,)

# Draw CDF
plt.subplot(1, 2, 2) 
for i, alpha in enumerate(alphas):
    color = colors[i]  # get corresponding color

    # Gamma distribution
    gamma_cdf = gamma.cdf(x, a=alpha, scale=1)

    # Log-normal distribution parameters
    sigma = np.sqrt(np.log(1 + 1/alpha))
    mu = np.log(alpha) - 0.5 * sigma**2
    lognormal_cdf = lognorm.cdf(x, s=sigma, scale=np.exp(mu))

    # plot CDF
    plt.plot(x, gamma_cdf, label=f'Gamma(α={alpha})', linestyle='--', color=color)
    plt.plot(x, lognormal_cdf, label=f'Log-normal(α={alpha})', color=color)

plt.title('Cumulative Distribution Functions (CDF)')
plt.xlabel('x')
plt.ylabel('Cumulative Probability')
plt.xlim(0, 4)

# Add legend in the bottom right corner
plt.legend(loc='lower right')

plt.tight_layout()

save_path = os.path.join(graph_dir, 'Moment matching.pdf')
plt.savefig(save_path, format="pdf")

plt.show()
