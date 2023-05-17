""" Script to compute optimal left candidate position as a function of gamma
"""
import logging
import sys
from functions import *

logging.basicConfig(level=logging.INFO)

def main(L, R, alpha, beta):
    """ Returns dataframe of left candidate final positions

    Input:
        L: (float) left candidate position
        R: (float) right candidate position, L < R.
        alpha: (float) left candidate opportunism.
        beta: (float) right candidate opportunism

    Output:
        Returns data frame with columns "gamma" and "final left position"
        which gives the optimal final position of the left candidate 
        for a given value of gamma.

    """
    means = np.array([-1.0,1.0])
    variances = np.array([0.5, 0.5])
    weights = np.array([0.5, 0.5])

    # Compute relevant linspace for left-right spectrum
    a = np.argmin(means)
    b = np.argmax(means)
    m = math.floor(means[a] - 3 * (variances[a] ** (1/2)))
    M = math.ceil(means[b] + 3 * (variances[b] ** (1/2)))

    # Place relevant political positions along spectrum
    n_positions = 100
    positions = np.linspace(m,M, n_positions)
    df = pd.DataFrame(positions, columns = ["Position"])

    # Compute probability density
    model = get_gaussian_mixture_model(means, variances, weights)
    x = df["Position"].values.reshape(-1,1)
    df["Density"] = pdf(x,model)

    # Select gamma for function input.
    gamma = np.linspace(2.5, 3.5, 20)
    final_ell_positions = []
    ell_dict = {}
    r_dict = {}
    logging.info(("...working..."))
    for i in range(len(gamma)):

        ell_positions, r_positions = coalescing_candidates_with_g(left_position = L, 
                                                        right_position = R, 
                                                        model = model, 
                                                        gamma = gamma[i], 
                                                        alpha = alpha, 
                                                        beta = beta)
        ell_dict[gamma[i]] = ell_positions
        r_dict[gamma[i]] = r_positions
        final_ell_positions.append(ell_positions[-1])

    df = pd.DataFrame()
    df["gamma"] = gamma
    df["final_left_position"] = final_ell_positions
    df.to_csv("discontinuity.csv")
    logging.info(f"Output printed to: discontinuity.csv")

if __name__ == '__main__':
    """ Optional input arguments L, R, alpha, beta
    """
    L = -1
    R = 2
    alpha = 1
    beta = 0.2
    if len(sys.argv) > 1:
        L, R, alpha, beta = sys.argv[1:]
    main(L, R, alpha, beta)
