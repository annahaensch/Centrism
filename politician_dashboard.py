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




# mandatory = st.radio(
#     "Do you think voting should be mandatory?",
#     ('Yes','No'))

# with open('survey.txt', 'r') as file:
#   lines = file.readlines()
#   if mandatory == "Yes":
#     yes = int(lines[0].split(",")[0]) + 1
#     no = int(lines[0].split(",")[1]) 
#   if mandatory == "No":
#     yes = int(lines[0].split(",")[0]) 
#     no = int(lines[0].split(",")[1]) + 1

#   yes_post = int(lines[1].split(",")[0])
#   no_post = int(lines[1].split(",")[1])

# lines[0] = f"{yes},{no}\n"

# # and write everything back
# with open('survey.txt', 'w') as file:
#     file.writelines(lines)

st.subheader("I. The Range of Voter Beliefs")

st.markdown("We'll start by seeding some populations with initial belief \n"+
        "distributions. We'll include a unimodal Gaussian centered at 0, \n"+
        "and a multimodal Gaussian whose parameters you can choose \n"+
        "in the sidebar on the left.")


# Place relevant political positions along left-right spectrum.
n_positions = 1000
positions = np.linspace(-5,5, n_positions)
df = pd.DataFrame(positions, columns = ["Position"])

# Select GMM parameters.
st.sidebar.subheader("Gaussian Mixture Model Parameters")
st.sidebar.markdown("Select the **means** for your Gaussian mixture model.  \n"+
        "The number of means that you choose will be the number of components \n"+
        "of your Gaussian mixture model.")
means = sidebar_text_field(r"$\mu$ = ", value= "-1, 1")
means = np.array([float(m.strip(" ")) for m in means.split(",")])

st.sidebar.markdown("Select the **variances** for your Gaussian mixture model.  \n"+
        "The number of variances that you choose should be the same as the \n"+
        "number of means you chose above.")
variances = sidebar_text_field(r"$\sigma$ = ", value= ".5, .5")
variances = np.array([float(s.strip(" ")) for s in variances.split(",")])

def get_gaussian_mixture_model(means, variances):
  """ Initialize multimodal model with chosen parameters."""
  n_components = means.shape[0]
  gmm = GaussianMixture(n_components=n_components, random_state=0)
  gmm.means_ = np.array([[m] for m in means])
  gmm.covariances_ = np.array([[[s]] for s in variances])
  gmm.weights_ = np.full(n_components,1/n_components)
  gmm.precision_ = np.zeros(gmm.covariances_.shape)
  gmm.precisions_cholesky_ = np.zeros(gmm.covariances_.shape)
  for i in range(gmm.precision_.shape[0]):
    gmm.precision_[i] = np.linalg.inv(gmm.covariances_[i])
    gmm.precisions_cholesky_[i] = np.linalg.cholesky(gmm.precision_[i]) ** 2

  return gmm

def pdf(x, model):
  """compute probability density function."""
  if type(x) == pd.Series:
    x = x.values.reshape(-1,1)
  elif type(x) == np.ndarray:
    x = x.reshape(-1,1)
  elif type(x) == list:
    x = np.array(x).reshape(-1,1)

  return np.exp(model.score_samples(x))

unimodal_model = get_gaussian_mixture_model(np.array([0.]), np.array([1.]))
multimodal_model = get_gaussian_mixture_model(means, variances)

df["Unimodal Distribution"] = pdf(df["Position"],unimodal_model)
df["Multimodal Distribution"] = pdf(df["Position"],multimodal_model)


chart_data_wide = pd.melt(df.reset_index(), id_vars=["Position"], 
  value_vars = ["Unimodal Distribution","Multimodal Distribution"])

chart_data_wide.rename(columns = {"variable":"Underlying Population",
                        "value":"Density"}, inplace = True)

chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Position", type = "quantitative"),
    y=alt.Y("Density", type = "quantitative"),
    color=alt.Color('Underlying Population',scale={"range": [left_blue, left_blue_light]}),
    strokeWidth = alt.value(4)
    )

st.altair_chart(chart, use_container_width=True)

st.subheader("II. A Simple Election")

st.markdown("Let's simulate a simple election where the winning candidate \n"+
            "is the one with the largest share of the electorate 'under the curve.' \n"+
            "In the bar charts below, the winning candidate will be highlighted.")

def pdf_integrand(x = df["Position"], model = multimodal_model):
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
    -5., 5., (-1.,1.8), step = 0.25
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


left_share = get_left_share(left_position = ell, right_position = r, model = multimodal_model)
right_share = get_right_share(left_position = ell, right_position = r, model = multimodal_model)
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
    title='Vote Share with a Multimodal Electorate'
)
)
st.altair_chart(chart, use_container_width=True)

st.markdown("To see this more clearly, we can look at the area under the curve \n"+
            "of the electorate density function for the multimodal distribution. \n"+
            "Notice how the winning candidate takes more than half of the area \n"+
            "under the curve.")

midpoint = (ell + r)/2
df_left = df[df["Position"] <= midpoint]
df_right = df[df["Position"] > midpoint]

chart_data_wide_left = pd.melt(df_left.reset_index(), id_vars=["Position"], 
  value_vars = ["Multimodal Distribution"])
chart_data_wide_right = pd.melt(df_right.reset_index(), id_vars=["Position"], 
  value_vars = ["Multimodal Distribution"])

chart_data_wide_left.rename(columns = {"variable":"Underlying Population",
                        "value":"Density"}, inplace = True)
chart_data_wide_right.rename(columns = {"variable":"Underlying Population",
                        "value":"Density"}, inplace = True)

chart_left= alt.Chart(chart_data_wide_left).mark_area().encode(
    x=alt.X("Position", type = "quantitative"),
    y=alt.Y("Density", type = "quantitative"),
    color =  alt.value(left_color),
    strokeWidth = alt.value(4)
    )

chart_right= alt.Chart(chart_data_wide_right).mark_area().encode(
    x=alt.X("Position", type = "quantitative"),
    y=alt.Y("Density", type = "quantitative"),
    color = alt.value(right_color),
    strokeWidth = alt.value(4)
    )


st.altair_chart(chart_left + chart_right, use_container_width=True)

f_text = r"If the right candidate has a fixed position, in this case $r$ = "+f"{r}, "
text = "then it looks like it's always in the best interest of the left candidate to move closer to the right candidate. To see this more clearly, we will compute the vote share as a function of the left candidate position, $\ell$, for a fixed $r$."

st.markdown(f_text + text)

ell_positions = np.linspace(-5,r)
left_shares_unimodal = []
left_shares_multimodal = []
for l in ell_positions:
  left_share_unimodal = get_left_share(left_position = l, right_position = r, model = unimodal_model)
  left_shares_unimodal.append(left_share_unimodal)
  left_share_multimodal = get_left_share(left_position = l, right_position = r, model = multimodal_model)
  left_shares_multimodal.append(left_share_multimodal)

chart_data = pd.DataFrame({"Position":ell_positions,
                      "Multimodal Distribution":left_shares_unimodal,
                      "Unimodal Distribution":left_shares_multimodal})

chart_data_wide = pd.melt(chart_data.reset_index(), id_vars=["Position"], 
  value_vars = ["Unimodal Distribution","Multimodal Distribution"])

chart_data_wide.rename(columns = {"variable":"Underlying Population",
                        "value":"Vote Share"}, inplace = True)

chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Position", type="quantitative", title="Left candidate position"),
    y=alt.Y("Vote Share", 
            type = "quantitative", 
            title = "Left candidate vote share",
            axis = alt.Axis(values = [0, 0.5, 1], grid = True)),
    color=alt.Color('Underlying Population',scale={"range": [left_blue, left_blue_light]}),
    strokeWidth = alt.value(4)
    ).properties(
        title='Left Candidate Vote Share as A Function of Position'
    )

line = alt.Chart(pd.DataFrame({'Vote Share': [0.5,1], "color":["white","black"]})).mark_rule(strokeDash=[5, 10]).encode(
                    y='Vote Share',
                    color = alt.Color('color:N', scale=None, legend=None))

st.altair_chart((chart+ line).resolve_scale(color='independent'), use_container_width=True)

st.markdown("As soon as the left candidate crosses the dashed line, they "+
            "have more than 50% of the votes and therefore they have won the election.")

st.subheader("II. Opportunistic Candidates")

# Add an alpha slider to the sidebar:
alpha_text = st.markdown("**1. Choose your &alpha; value.**")
alpha= st.slider(
    r"This is a measure of the left-wing candidate's eagerness. A greater value of $\alpha$, means the left-wing candidtae will move more eagerly (i.e. rapidly) towards the opportunistic position.",
    0.0, 2.0, (1.5)
)

# Add a beta slider to the sidebar:
beta_text = st.markdown("**2. Choose your &beta; value.**")
beta = st.slider(
    r"This is a measure of the right-wing candidate's eagerness. A greater value of $\beta$, means the right-wing candidtae will move more eagerly (i.e. rapidly) towards the opportunistic position.",
    0.0, 2.0, (1.0)
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
    delta = pdf(x = [(left_position + right_position)/2], model = multimodal_model)[0]
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
                                                  model = multimodal_model, 
                                                  alpha = alpha, 
                                                  beta = beta)

chart_data = pd.DataFrame({"Left":ell_positions,
                      "Right":r_positions})

chart_data_wide = pd.melt(chart_data.reset_index(), id_vars=["index"], 
  value_vars = ["Left","Right"])
chart_data_wide.rename(columns = {"variable":"Candidate"}, inplace = True)

chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("index", type="quantitative", title="Timestep"),
    y=alt.Y("value", type = "quantitative", title = "Candidate position"),
    color=alt.Color('Candidate',scale={"range": [left_blue, right_red]}),
    strokeDash='Candidate',
    strokeWidth = alt.value(4)
    ).properties(
        title='Candidates Moving Accoring to Steepest Ascent Will Eventually Meet'
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
	left_share = left_share_with_g(left_position = l, right_position = r, gamma = gamma, model = multimodal_model)
	left_shares.append(left_share)
chart_data["Multimodal Distribution"] = left_shares

chart_data_wide = pd.melt(chart_data, id_vars=["Left"], 
  value_vars = ["Unimodal Distribution","Multimodal Distribution"])
chart_data_wide.rename(columns = {"variable":"Underlying Population",
                        "value":"Left Candidate Vote Share",
                        "Left":"Candidate Position"}, inplace = True)

chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Candidate Position", 
            type="quantitative", 
            title="Left candidate position"),
    y=alt.Y("Left Candidate Vote Share", 
            type = "quantitative", 
            title = "Left candidate vote share",
            axis = alt.Axis(values = [0, 0.5, 1], grid = True)),
    color=alt.Color('Underlying Population',scale={"range": [left_blue, left_blue_light]}),
    strokeWidth = alt.value(4)
    ).properties(
        title='Left Candidate Vote Share as A Function of Position with Opportunism'
    )

line = alt.Chart(pd.DataFrame({'Vote Share': [0.5,1], "color":["white","black"]})).mark_rule(strokeDash=[5, 10]).encode(
                    y='Vote Share',
                    color = alt.Color('color:N', scale=None, legend=None))

st.altair_chart((chart+ line).resolve_scale(color='independent'), use_container_width=True)

st.markdown("Once again, as soon as the left candidate crosses the dashed line, they "+
            "have more than 50% of the votes and therefore they have won the election.")


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
left_derivatives_multimodal = []
for l in ell_positions:
  D_unimodal = left_derivative(l,right_position = r, gamma = gamma, model = unimodal_model)
  left_derivatives_unimodal.append(D_unimodal)
  D_multimodal = left_derivative(l,right_position = r, gamma = gamma, model = multimodal_model)
  left_derivatives_multimodal.append(D_multimodal)

chart_data["Unimodal Distribution"] = left_derivatives_unimodal
chart_data["Multimodal Distribution"] = left_derivatives_multimodal

chart_data_wide = pd.melt(chart_data, id_vars=["Left"], 
  value_vars = ["Unimodal Distribution","Multimodal Distribution"])
chart_data_wide.rename(columns = {"variable":"Underlying Population",
                        "value":"Rate of Change in Vote Share",
                        "Left":"Candidate Position"}, inplace = True)

chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Candidate Position", type="quantitative", title="Left candidate position"),
    y=alt.Y("Rate of Change in Vote Share", type = "quantitative", title = "Rate of change in vote share"),
    color=alt.Color('Underlying Population',scale={"range": [left_blue, left_blue_light]}),
    strokeWidth = alt.value(4)
    ).properties(
        title='Rate of Change in Left Candidate Vote Share as A Function of Position'
    )

line = alt.Chart(pd.DataFrame({'Rate of Change': [0], "color":["white"]})).mark_rule(
                strokeDash=[5, 10]).encode(
                    y='Rate of Change',
                    color = alt.Color('color:N', scale=None, legend=None))

st.altair_chart((chart+ line).resolve_scale(color='independent'), use_container_width=True)

st.markdown("You can learn more about the mathematics behind this app in this paper: \n"+
  "Börgers, Christoph, Bruce Boghosian, Natasa Dragovic, and Anna Haensch. \n"+
  "_A blue sky bifurcation in the dynamics of political candidates._ \n"+
  "[arXiv preprint arXiv:2302.07993](https://arxiv.org/abs/2302.07993) (2023).")

# mandatory = st.radio(
#     "Now we'll ask again: do you think voting should be mandatory?",
#     ('Yes','No'))

# with open('survey.txt', 'r') as file:
#   lines = file.readlines()
#   if mandatory == "Yes":
#     yes = int(lines[0].split(",")[0]) + 1
#     no = int(lines[0].split(",")[1]) 
#   if mandatory == "No":
#     yes = int(lines[0].split(",")[0]) 
#     no = int(lines[0].split(",")[1]) + 1

#   yes_post = int(lines[1].split(",")[0])
#   no_post = int(lines[1].split(",")[1])

# lines[0] = f"{yes},{no}\n"

# # and write everything back
# with open('survey.txt', 'w') as file:
#     file.writelines(lines)