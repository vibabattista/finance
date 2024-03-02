import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm

# Stock price parameters
S0 = 100
mu = 0.1
sigmas = [0.4,0.6,0.8,1, 1.2, 1.5, 1.75, 2]
k1 = 105
k2 = 95
r = 0.05
T = 10
confidence = 0.95
h = 1
n = 1_000_000
for sigma in sigmas:
    # Option prices at time 0
    d1 = lambda K, S0, T: (np.log(S0/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    c0 = S0*norm.cdf(d1(k1, S0, T)) - k1*np.exp(-r*T)*norm.cdf(d1(k1, S0, T)-sigma*np.sqrt(T))
    p0 = k2*np.exp(-r*T)*norm.cdf(-d1(k2, S0, T)+sigma*np.sqrt(T)) - S0*norm.cdf(-d1(k2, S0, T))
    v0 = p0 - c0

    # Computing the prices after h years
    T = T - h
    Zt = np.random.normal(size=n)
    St = S0*np.exp((mu-0.5*sigma**2)*h + Zt*sigma*np.sqrt(h))

    # Call and put prices at time h in each scenario
    ct = St*norm.cdf(d1(k1, St, T)) - k1*np.exp(-r*T)*norm.cdf(d1(k1, St, T)-sigma*np.sqrt(T))
    pt = k2*np.exp(-r*T)*norm.cdf(-d1(k2, St, T)+sigma*np.sqrt(T)) - St*norm.cdf(-d1(k2, St, T))

    # Value of straddle in each scenario
    vt = pt - ct

    # Loss calculation
    vvar = v0 - vt
    vvar = np.sort(vvar)

    # Extracting VaR at the right confidence level from the loss
    ivar = round((confidence)*n)
    var = vvar[ivar]

    # Calculating ES
    ESv = np.mean(vvar[ivar:n])

    print("VaR:", var)
    print("ES:", ESv)
    # Subplot for each sigma value
    # Histogram of the loss
    # Subplot for each sigma value
    plt.subplot(4, 2, sigmas.index(sigma) + 1)
    plt.hist(vvar, bins=10000, density = True)
    plt.xlim(-200, 600)
    plt.ylim(0, 0.015)
    plt.title(f"Sigma: {sigma}")

plt.tight_layout()
plt.show()
