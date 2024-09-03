import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

def generalized_normal_pdf(x, mu, alpha, beta):
    """
    Calculate the probability density function of the generalized normal distribution.
    
    Parameters:
    - x: Values at which to calculate the PDF
    - mu: Location parameter
    - alpha: Scale parameter
    - beta: Shape parameter
    
    Returns:
    - PDF values
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive values")
    
    pdf = (beta / (2 * alpha * gamma(1/beta))) * np.exp(-((np.abs(x - mu) / alpha) ** beta))
    return pdf

# Range of values
x_values = np.linspace(-5, 5, 1000)

# Set parameters
mu = 0
alpha = 1

# Set different values for beta
beta_values = [2,1,0.5]

# Plot for each beta
for beta in beta_values:
    pdf_values = generalized_normal_pdf(x_values, mu, alpha, beta)
    plt.plot(x_values, pdf_values, label='Beta = {}'.format(beta))

plt.title('Generalized Normal Distribution for Different Beta Values')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()

# Save the plot as an image file (e.g., PNG)
plt.savefig('generalized_normal_distribution_beta_variation.png')
