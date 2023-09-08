""" Functions for politician_dashboard.py Streamlit App
"""
import math
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from scipy.integrate import quad

LEFT_BLUE = "#8a9dba" #"#446B84"
RIGHT_RED = "#f59585" #"#E58073"
RIGHT_RED_DARK = "#cc4e3e"
LEFT_BLUE_DARK = "#224f6b"
CENTRIST_GREY = "#a2a2a2" #"#E8E8E8"

def get_gaussian_mixture_model(means, variances, weights):
  """ Initialize multimodal model with chosen parameters.

  Inputs: 
    means: (array) float values for component means.
    variances: (array) float values for component variances

  Returns: 
    Fit GaussianMixtureModel.
  """
  n_components = means.shape[0]
  gmm = GaussianMixture(n_components=n_components, random_state=0)
  gmm.means_ = np.array([[m] for m in means])
  gmm.covariances_ = np.array([[[s]] for s in variances])
  gmm.weights_ = np.array([s for s in weights])
  gmm.precision_ = np.zeros(gmm.covariances_.shape)
  gmm.precisions_cholesky_ = np.zeros(gmm.covariances_.shape)
  for i in range(gmm.precision_.shape[0]):
    gmm.precision_[i] = np.linalg.inv(gmm.covariances_[i])
    gmm.precisions_cholesky_[i] = np.linalg.cholesky(gmm.precision_[i]) ** 2

  return gmm

def pdf(x, model):
  """compute probability density function.

  Inputs: 
    x: (array) positions from left to right.
    model: (GaussiaMixtureModel) fit model.

  Returns:
    Probability density function for model evaluated at x.
  """
  if type(x) == pd.Series:
    x = x.values.reshape(-1,1)
  elif type(x) == np.ndarray:
    x = x.reshape(-1,1)
  elif type(x) == list:
    x = np.array(x).reshape(-1,1)

  return np.exp(model.score_samples(x))



def get_left_share(df, left_position, right_position, model):
  """Compute left candidate vote share."""
  lower = -np.inf
  upper = (left_position + right_position)/2
  
  def pdf_integrand(x = df["Position"], model = model):
      return pdf([x], model)

  return quad(pdf_integrand, lower, upper, args=(model))[0]

def get_right_share(df, left_position, right_position, model):
  """Compute right candidate vote share."""
  lower = (left_position + right_position)/2
  upper = np.inf

  def pdf_integrand(x = df["Position"], model = model):
      return pdf([x], model)

  return quad(pdf_integrand, lower, upper, args=(model))[0]

def get_plot_colors(left_share, right_share):
  """ Get plot colors based on winner."""
  left_color = CENTRIST_GREY
  right_color = CENTRIST_GREY

  if left_share > right_share:
    left_color = LEFT_BLUE
    right_color = CENTRIST_GREY

  if left_share < right_share:
    left_color = CENTRIST_GREY
    right_color = RIGHT_RED
  return left_color, right_color

def get_intersection(a1, a2, b1, b2):
  """ Compute intersection of two lines.
          y = (a2 - a1)x + a1
          y = (b2 - b1)x + b1
  """
  x = (a1 - b1) /  ((b2 - b1) - (a2 - a1))
  y = ((a2 - a1) * x) + a1
  return x,y

def coalescing_candidates(left_position, right_position, model, alpha = 1, 
                      beta = .5):
  """Simulate colaescing of candidate positions.

  Inputs: 
    left_position: (float) position of left candidate.
    right_position: (float) position of right candidate.
    model: (GaussianMixtureModel) fit instance.
    alpha: (float) left candidate eagerness.
    beta: (float) right candidate eagerness.

  Returns: 
    Left and right candidate positions over time as the move according to 
    steepest ascent.
  """
  ell_positions = []
  r_positions = []
  y = None
  while left_position < right_position:
    delta = pdf(x = [(left_position + right_position)/2], model = model)[0]
    left_position += (alpha/2)*delta
    right_position += (-beta/2)*delta
    if left_position < right_position:
      ell_positions.append(left_position)
      r_positions.append(right_position) 
    else:
      x, y = get_intersection(a1 = ell_positions[-1], 
                              a2 = left_position, 
                              b1 = r_positions[-1], 
                              b2 = right_position)
      ell_positions.append(y)
      r_positions.append(y)

    if len(ell_positions) == 50:
      break

  if y:
    for i in range(10):
        ell_positions.append(y)
        r_positions.append(y)

  return ell_positions, r_positions

def g_func(z, gamma):
  return np.exp(-z/gamma)

def left_integrand(x, left_position, gamma, model):
    x = np.array([x])
    f = pdf(x, model = model)
    g = g_func(np.abs(left_position - x), gamma)
    return f * g

def right_integrand(x, right_position, gamma, model):
    x = np.array([x])
    f = pdf(x, model = model)
    g = g_func(np.abs(right_position - x), gamma)
    return f * g

def left_integral(left_position, right_position, gamma, model):
  """ Compute numerical integral for left candidate share."""
  lower = -np.inf
  upper = (left_position + right_position)/2
  I = quad(left_integrand, lower, upper, args=(left_position, gamma, model))
  return I[0]

def right_integral(left_position, right_position, gamma, model):
  """ Compute numerical integral for right candidate share."""
  lower = (left_position + right_position)/2
  upper = np.inf
  I = quad(right_integrand, lower, upper, args=(right_position, gamma, model))
  return I[0]

def left_share_with_g(left_position, right_position, gamma, model):
  """Compute left candidate vote share."""
  if left_position > right_position: 
    raise ValueError("Left candidate must be to left of right candidate.")
  return left_integral(left_position, right_position, gamma, model)

def right_share_with_g(left_position, right_position, gamma, model):
  """Compute left candidate vote share."""
  if left_position > right_position: 
    raise ValueError("Left candidate must be to left of right candidate.")
  return right_integral(left_position, right_position, gamma, model)

def left_derivative(x,right_position, gamma, model, h = 0.001):
  """Compute numerical derivative for left candidate change."""
  forward_sum = left_share_with_g(x + h, right_position, gamma, model)
  backward_sum = left_share_with_g(x - h, right_position, gamma, model)
  return (forward_sum - backward_sum) / (2*h)

def right_derivative(x,left_position, gamma, model, h = 0.001):
  """Compute numerical derivative for right candidate change."""
  forward_sum = right_share_with_g(left_position, x + h, gamma, model)
  backward_sum = right_share_with_g(left_position, x - h, gamma, model)
  return (forward_sum - backward_sum) / (2*h)

def coalescing_candidates_with_g(left_position, right_position, model, gamma, alpha, beta):
  """Simulate colaescing of candidate positions with voter tolerance."""
  ell_positions = []
  r_positions = []
  y = None
  for i in range(100):
    left_delta = left_derivative(left_position, right_position, gamma, model)
    right_delta = right_derivative(right_position, left_position, gamma, model)
    left_position += alpha*left_delta
    right_position += beta*right_delta
    if left_position < right_position:
      ell_positions.append(left_position)
      r_positions.append(right_position) 
    else:
      x, y = get_intersection(ell_positions[-1], left_position, r_positions[-1], right_position)
      ell_positions.append(y)
      r_positions.append(y)
      break

  if y:
    for j in range(i,100):
      ell_positions.append(y)
      r_positions.append(y)
      
  return ell_positions, r_positions