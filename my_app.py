""" Political Centrism Dashboard 

This code launches a streamlit app 

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.mixture import GaussianMixture
from scipy.integrate import quad

left_blue = "#446B84"
left_blue_light = "#a7bad6"
right_red = "#E58073"
centrist_grey = "#E8E8E8"

def sidebar_text_field(label, columns=None, **input_params):
    c1, c2 = st.sidebar.columns(columns or [1, 4])

    # Display field name with some alignment
    c1.markdown("##")
    c1.markdown(label)

    # Sets a default key parameter to avoid duplicate key errors
    input_params.setdefault("key", label)

    # Forward text input parameters
    return c2.text_input("", **input_params)


st.title("The Perils of Political Centrism")

st.subheader("I. The Range of Voter Beliefs")

st.markdown("We'll start by seeding two populations with initial belief \n"+
        "distributions. Some initial values have been set for you, \n"+
        "but you can update them in the sidebar on the left.")


# Place relevant political positions along left-right spectrum.
n_positions = 1000
positions = np.linspace(-5,5, n_positions)
df = pd.DataFrame(positions, columns = ["position"])

# Generate unimodal model.
st.sidebar.subheader("Unimodal Distribution Parameters")
mu = sidebar_text_field(r"$\mu$ = ", value= 0)
sigma = sidebar_text_field(r"$\sigma$ = ", value= 1)

unimodal_x = np.random.normal(loc = float(mu), scale = float(sigma), size = 1000).reshape(-1,1)
unimodal_model = GaussianMixture(n_components=1, random_state=0).fit(unimodal_x)

# Generate bimodal model.
st.sidebar.subheader("Bimodal Distribution Parameters")
mu_1 = sidebar_text_field(r"$\mu_\text{left}$ = ", value= -1)
sigma_1 = sidebar_text_field(r"$\sigma_\text{left}$ = ", value= .5)
mu_2 = sidebar_text_field(r"$\mu_\text{right}$ = ", value= 1)
sigma_2 = sidebar_text_field(r"$\sigma_\text{right}$ = ", value= .5)

x1 = np.random.normal(loc = float(mu_1), scale = float(sigma_1), size = 1000).reshape(-1,1)
x2 = np.random.normal(loc = float(mu_2), scale = float(sigma_2), size = 1000).reshape(-1,1)
bimodal_x = np.concatenate([x1,x2])

bimodal_model = GaussianMixture(n_components=2, random_state=0).fit(bimodal_x)
bimodal_model.means_ = np.array([[float(mu_1)],[float(mu_2)]])
bimodal_model.covariances_ = np.array([[[float(sigma_1)]],[[float(sigma_2)]]])
bimodal_model.weights_ = np.array([0.5, 0.5])

def pdf(x, model):
  """compute probability density function."""
  if type(x) == pd.Series:
    x = x.values.reshape(-1,1)
  elif type(x) == np.ndarray:
    x = x.reshape(-1,1)
  elif type(x) == list:
    x = np.array(x).reshape(-1,1)

  return np.exp(model.score_samples(x))

df["Unimodal Distribution"] = pdf(df["position"],unimodal_model)
df["Bimodal Distribution"] = pdf(df["position"],bimodal_model)

chart_data_wide = pd.melt(df.reset_index(), id_vars=["position"], 
  value_vars = ["Unimodal Distribution","Bimodal Distribution"])

chart_data_wide.rename(columns = {"variable":"Underlying Population",
                        "value":"density"}, inplace = True)

chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("position", type = "quantitative"),
    y=alt.Y("density", type = "quantitative"),
    color=alt.Color('Underlying Population',scale={"range": [left_blue, left_blue_light]}),
    strokeWidth = alt.value(4)
    )

st.altair_chart(chart, use_container_width=True)

st.subheader("II. A Simple Election")

st.markdown("First we'll simulate a simple election where the winning candidate \n"+
            "is the one with the largest share of the electorate 'under the curve.' \n"+
            "In the bar charts below, the winning candidate will be highlighted.")

def pdf_integrand(x = df["position"], model = bimodal_model):
  return pdf([x], model)

def get_left_share(left_position, right_position, model):
  """Compute left candidate vote share."""
  lower = -np.inf
  upper = (left_position + right_position)/2
  return quad(pdf_integrand, lower, upper, args=(model))[0]

def get_right_share(left_position, right_position, model):
  """Compute right candidate vote share."""
  lower = (left_position + right_position)/2
  upper = np.inf
  return quad(pdf_integrand, lower, upper, args=(model))[0]

left_right= st.slider(
    r"Choose the left and right candidate poisitions using this slider.",
    -5., 5., (-1.,3.), step = 0.25
)

ell = left_right[0]
r = left_right[1]

def get_plot_colors(left_share, right_share):
  """ Get plot colors based on winner."""
  left_color = centrist_grey
  right_color = centrist_grey

  if left_share > right_share:
    left_color = left_blue
    right_color = centrist_grey

  if left_share < right_share:
    left_color = centrist_grey
    right_color = right_red
  return left_color, right_color

left_share = get_left_share(left_position = ell, right_position = r, model = unimodal_model)
right_share = get_right_share(left_position = ell, right_position = r, model = unimodal_model)
left_color, right_color = get_plot_colors(left_share, right_share)

chart_data = pd.DataFrame([[left_share],[right_share]], 
          columns = ["share"],
          index = ["Left Candidate", "Right Candidate"])
chart_data_wide = pd.melt(chart_data.reset_index(), id_vars=["index"])

# Horizontal stacked bar chart
chart = (
    alt.Chart(chart_data_wide)
    .mark_bar()
    .encode(
        x=alt.X("value", type="quantitative", title=""),
        y=alt.Y("index", type="nominal", title=""),
        color = alt.condition(alt.datum.index == "Left Candidate", 
                alt.value(left_color),
                alt.value(right_color))
    ).properties(
    title='Vote Share with a Unimodal Electorate'
)
)
st.altair_chart(chart, use_container_width=True)

left_share = get_left_share(left_position = ell, right_position = r, model = bimodal_model)
right_share = get_right_share(left_position = ell, right_position = r, model = bimodal_model)
left_color, right_color = get_plot_colors(left_share, right_share)

chart_data = pd.DataFrame([[left_share],[right_share]], 
          columns = ["share"],
          index = ["Left Candidate", "Right Candidate"])
chart_data_wide = pd.melt(chart_data.reset_index(), id_vars=["index"])

# Horizontal stacked bar chart
chart = (
    alt.Chart(chart_data_wide)
    .mark_bar()
    .encode(
        x=alt.X("value", type="quantitative", title=""),
        y=alt.Y("index", type="nominal", title=""),
        color = alt.condition(alt.datum.index == "Left Candidate", 
                alt.value(left_color),
                alt.value(right_color))
    ).properties(
    title='Vote Share with a Bimodal Electorate'
)
)
st.altair_chart(chart, use_container_width=True)

st.markdown("If the right candidate has a fixed position, $r$, it looks like \n"+
            "it's always in the best interest of the left candidate to move \n"+ 
            "closer to the right candidate. To see this more clearly, we will \n"+
            "compute the vote share as a function of the left candidate \n"+
            "position, $\ell$, for a fixed $r$.")

ell_positions = np.linspace(-5,r)
left_shares_unimodal = []
left_shares_bimodal = []
for l in ell_positions:
  left_share_unimodal = get_left_share(left_position = l, right_position = r, model = unimodal_model)
  left_shares_unimodal.append(left_share_unimodal)
  left_share_bimodal = get_left_share(left_position = l, right_position = r, model = bimodal_model)
  left_shares_bimodal.append(left_share_bimodal)

chart_data = pd.DataFrame({"position":ell_positions,
                      "Bimodal Distribution":left_shares_unimodal,
                      "Unimodal Distribution":left_shares_bimodal})

chart_data_wide = pd.melt(chart_data.reset_index(), id_vars=["position"], 
  value_vars = ["Unimodal Distribution","Bimodal Distribution"])

chart_data_wide.rename(columns = {"variable":"Underlying Population",
                        "value":"Vote Share"}, inplace = True)

chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("position", type="quantitative", title="Left Candidate Position"),
    y=alt.Y("Vote Share", type = "quantitative", title = "Left Candidate Vote Share"),
    color=alt.Color('Underlying Population',scale={"range": [left_blue, left_blue_light]}),
    strokeWidth = alt.value(4)
    ).properties(
        title='Left Candidate Vote Share as A Function of Position'
    )
st.altair_chart(chart, use_container_width=True)

st.subheader("II. Opportunistic Candidates")

# Add an alpha slider to the sidebar:
alpha_text = st.markdown("**1. Choose your &alpha; value.**")
alpha= st.slider(
    r"This is a measure of the left-wing candidate's eagerness. A greater value of $\alpha$, means the left-wing candidtae will move more eagerly (i.e. rapidly) towards the opportunistic position.",
    0.0, 5.0, (2.0)
)

# Add a beta slider to the sidebar:
beta_text = st.markdown("**2. Choose your &beta; value.**")
beta = st.slider(
    r"This is a measure of the right-wing candidate's eagerness. A greater value of $\beta$, means the right-wing candidtae will move more eagerly (i.e. rapidly) towards the opportunistic position.",
    0.0, 5.0, (3.0)
)

def get_intersection(a1, a2, b1, b2):
  """ Compute intersection of two lines.
          y = (a2 - a1)x + a1
          y = (b2 - b1)x + b1
  """
  x = (a1 - b1) /  ((b2 - b1) - (a2 - a1))
  y = ((a2 - a1) * x) + a1
  return x,y

def coalescing_candidates(left_position, right_position, model, alpha = 1, beta = .5):
  """Simulate colaescing of candidate positions."""
  ell_positions = []
  r_positions = []
  y = None
  while left_position < right_position:
    delta = pdf(x = [(left_position + right_position)/2], model = bimodal_model)[0]
    left_position += (alpha/2)*delta
    right_position += (-beta/2)*delta
    if left_position < right_position:
      ell_positions.append(left_position)
      r_positions.append(right_position) 
    else:
      x, y = get_intersection(ell_positions[-1], left_position, r_positions[-1], right_position)
      ell_positions.append(y)
      r_positions.append(y)

  if y:
    for i in range(10):
        ell_positions.append(y)
        r_positions.append(y)

  return ell_positions, r_positions


ell_positions, r_positions = coalescing_candidates(left_position = ell, 
                                                  right_position = r, 
                                                  model = bimodal_model, 
                                                  alpha = alpha, 
                                                  beta = beta)

chart_data = pd.DataFrame({"Left":ell_positions,
                      "Right":r_positions})

chart_data_wide = pd.melt(chart_data.reset_index(), id_vars=["index"], 
  value_vars = ["Left","Right"])
chart_data_wide.rename(columns = {"variable":"Candidate"}, inplace = True)

chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("index", type="quantitative", title="Timestep"),
    y=alt.Y("value", type = "quantitative", title = "Candidate Position"),
    color=alt.Color('Candidate',scale={"range": [left_blue, right_red]}),
    strokeDash='Candidate',
    strokeWidth = alt.value(4)
    ).properties(
        title='Candidates Moving Accoring to Steepest Ascent Will Eventaully Meet'
    )
st.altair_chart(chart, use_container_width=True)


st.subheader("III. Loyal Voters")

# Add a gamma slider to the sidebar:
gamma_text = st.markdown("**3. Choose your &gamma; value.**")
gamma = st.slider(
     	r'This is a measure of voter loyalty. A greater value of $\gamma$, means the voter is more likely to stick with the candidate, even as their position drifts.',
    	0.0, 10.0, (5.0)
)

def g_func(z, gamma):
  return np.exp(-z/gamma)

def integrand(x, left_position, right_position, gamma, model):
    x = np.array([x])
    f = pdf(x, model = model)
    g = g_func(np.abs(left_position - x), gamma)
    return f * g

def left_integral(left_position, right_position, gamma, model):
  """ Compute numerical integral for left candidate share."""
  lower = -np.inf
  upper = (left_position + right_position)/2
  I = quad(integrand, lower, upper, args=(left_position, right_position, gamma, model))
  return I[0]

def right_integral(left_position, right_position, gamma, model):
  """ Compute numerical integral for right candidate share."""
  lower = (left_position + right_position)/2
  upper = np.inf
  I = quad(integrand, lower, upper, args=(left_position, right_position, gamma, model))
  return I[0]

def left_share_with_g(left_position, right_position, gamma, model):
  """Compute left candidate vote share."""
  if left_position > right_position: 
    raise ValueError("The left candidate must be to the left of the right candidate.")
  return left_integral(left_position, right_position, gamma, model)

def right_share_with_g(left_position, right_position, gamma, model):
  """Compute left candidate vote share."""
  if left_position > right_position: 
    raise ValueError("The left candidate must be to the left of the right candidate.")
  return right_integral(left_position, right_position, gamma, model)


ell_positions = np.linspace(-5,r)
chart_data = pd.DataFrame()
chart_data["Left"] = ell_positions

left_shares = []
left_positions = []
for l in ell_positions:
	left_share = left_share_with_g(left_position = l, right_position = r, gamma = gamma, model = unimodal_model)
	left_shares.append(left_share)
chart_data["Unimodal Distribution"] = left_shares

left_shares = []
left_positions = []
for l in ell_positions:
	left_share = left_share_with_g(left_position = l, right_position = r, gamma = gamma, model = bimodal_model)
	left_shares.append(left_share)
chart_data["Bimodal Distribution"] = left_shares

chart_data_wide = pd.melt(chart_data, id_vars=["Left"], 
  value_vars = ["Unimodal Distribution","Bimodal Distribution"])
chart_data_wide.rename(columns = {"variable":"Underlying Population",
                        "value":"Vote Share",
                        "Left":"Candidate Position"}, inplace = True)

chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Candidate Position", type="quantitative", title="Position"),
    y=alt.Y("Vote Share", type = "quantitative", title = "Vote Share"),
    color=alt.Color('Underlying Population',scale={"range": [left_blue, left_blue_light]}),
    strokeWidth = alt.value(4)
    ).properties(
        title='Left Candidate Vote Share as A Function of Position'
    )
st.altair_chart(chart, use_container_width=True)

st.markdown("Next, we will look at how the share of votes changes as a function "+
            "of candidate position, with the introduction of the intolerant voter "+
            "function.  We'll define the following two functions to help us "+
            "compute numerical derivatives.  We'll be approximating derivatives "+
            "using the symmetric difference quotient.")

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


ell_positions = np.linspace(-5,r -.001)
chart_data = pd.DataFrame()
chart_data["Left"] = ell_positions
left_derivatives_unimodal = []
left_derivatives_bimodal = []
for l in ell_positions:
  D_unimodal = left_derivative(l,right_position = r, gamma = gamma, model = unimodal_model)
  left_derivatives_unimodal.append(D_unimodal)
  D_bimodal = left_derivative(l,right_position = r, gamma = gamma, model = bimodal_model)
  left_derivatives_bimodal.append(D_bimodal)

chart_data["Unimodal Distribution"] = left_derivatives_unimodal
chart_data["Bimodal Distribution"] = left_derivatives_bimodal

chart_data_wide = pd.melt(chart_data, id_vars=["Left"], 
  value_vars = ["Unimodal Distribution","Bimodal Distribution"])
chart_data_wide.rename(columns = {"variable":"Underlying Population",
                        "value":"Change in Vote Share",
                        "Left":"Candidate Position"}, inplace = True)

st.text(chart_data_wide)
chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Candidate Position", type="quantitative", title="Position"),
    y=alt.Y("Change in Vote Share", type = "quantitative", title = "Change in Vote Share"),
    color=alt.Color('Underlying Population',scale={"range": [left_blue, left_blue_light]}),
    strokeWidth = alt.value(4)
    ).properties(
        title='Change in Left Candidate Vote Share as A Function of Position'
    )
st.altair_chart(chart, use_container_width=True)