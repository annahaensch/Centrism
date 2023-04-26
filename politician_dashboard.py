""" Political Centrism Dashboard 

This code launches a streamlit app 

"""
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.mixture import GaussianMixture
from scipy.integrate import quad

from load_css import local_css

LEFT_BLUE = "#8a9dba" #"#446B84"
RIGHT_RED = "#f59585" #"#E58073"
RIGHT_RED_DARK = "#cc4e3e"
LEFT_BLUE_DARK = "#224f6b"
CENTRIST_GREY = "#E8E8E8"

local_css("style.css")

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

st.title("ODEs and Mandatory Voting")

st.markdown("TODO: Add some text about reading the paper...introduce ideas.")

############## Section I #######################
st.header("I. The Distribution of Voter Beliefs")
#################################################

st.markdown("""We'll start by seeding a population with political beliefs on a 
  left-right spectrum. To keep things simple, let's assume that beliefs are 
  sampled from a multimodal Gaussian distribution.  One feature of this kind of 
  distribution is that it splits a population of belief holders into several 
  subpopulations with their beliefs clustered around different means. 
 """)

q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
        <span class='highlight red'>
          <span class='bold'>
            Discussion Question: <br>
          </span>
        </span>  
        <span class='highlight grey'>
          Is this sort of distribution a reasonable one for the population of US voters? 
        </span>
      </div>
    """

st.markdown(q,unsafe_allow_html=True)

st.markdown(r"""
  Suppose we agree that it's reasonable and assume our population 
  has beliefs that are sampled from an mixture of $M$ Gaussians where the $m^{th}$ 
  mode has mean $\mu_m$, variance $\sigma_m^2$ and weight $\omega_m$. Then the
  probability of a voter holding belief $x$ in the $m^{th}$ mode is sampled as 
  """)

st.latex(r'''
    p(x\mid y = m)  \sim \mathcal{N}(\mu_m, \sigma^2_m),
    ''')

st.markdown("""and the density of voters at position $x$ is given by """)

st.latex(r'''
    f(x)  = \sum_{m=1}^M \omega_m\cdot \frac{1}{\sigma_m\sqrt{2\pi}}e^{
    -\frac{1}{2}\left(\frac{x - \mu_m}{\sigma_m}\right)^2}.
    ''')

#TODO: Change title.

st.markdown("""In the boxes below enter the means and variances that you'd like 
  to use for each of the Gaussian modes.""")

# Input means and variances as strings
c1, c2, c3 = st.columns(3)
means = c1.text_input(label = "means", value = "-1,1")
variances = c2.text_input(label = "variances", value = ".5, .5")
weights = c3.text_input(label = "weights", value = ".5, .5")

# Get means and variances as floats
means = np.array([float(m.strip(" ")) for m in means.split(",")])
variances = np.array([float(s.strip(" ")) for s in variances.split(",")])
weights = np.array([float(w.strip(" ")) for w in weights.split(",")])
weights = weights/ np.sum(weights)

msg = """You have to have the same number of means and variances.  
  One for each mode."""
assert len(means) == len(variances), msg

msg = """You have to have the same number of means and weights.  
  One for each mode."""
assert len(means) == len(weights), msg

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

# Get chart data
chart_data_wide = pd.melt(df.reset_index(), 
                          id_vars=["Position"], 
                          value_vars = ["Density"])
chart_data_wide.rename(columns = {"value":"Density"}, inplace = True)

# Make chart
chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Position", type = "quantitative"),
    y=alt.Y("Density", type = "quantitative"),
    color=alt.value(LEFT_BLUE),
    strokeWidth = alt.value(4)
    ).properties(title='Density of Beliefs on the Left-Right Spectrum')

st.altair_chart(chart, use_container_width=True)


############## Section II #######################
st.header("II. When Voting is Mandatory")
#################################################

st.subheader("A. Static Candidates")
st.markdown(r"""Let's simulate a simple election in which everybody votes.  Supppose 
  that the left candidate, $L$, is in position $\ell$ and the right candidate, 
  $R$, is in position $r$.  The left candidate gets all of the votes 
  to their left and every vote up until the midway point between the $L$ and $R$.
  The share of votes belonging to the left and right candidates, $S_L$ and $S_R$,
  respectively, can also be framed in terms of __area under the curve__, as """)

st.latex(r'''
  S_L = \int_{-\infty}^{\frac{\ell+r}{2}} f(x) \, \, dx \hspace{1cm}\text{ and } 
  \hspace{1cm} S_R = \int_{\frac{\ell+r}{2}}^{\infty} f(x) \, \, dx
  ''')

st.markdown("""Using the slider below, choose your left and right candidate 
  positions.  Based on the positions you choose, the vote shares will be shown 
  in the chart that follows.""")

L, R = st.slider("",float(m), float(M), (-1.,1.8), step = 0.25)


def pdf_integrand(x = df["Position"], model = model):
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

left_share = get_left_share(left_position = L, right_position = R, model = model)
right_share = get_right_share(left_position = L, right_position = R, model = model)
left_color, right_color = get_plot_colors(left_share, right_share)

# Get chart data
chart_data = pd.DataFrame([[left_share],[right_share]], 
          columns = ["share"],
          index = ["Left Candidate", "Right Candidate"])
chart_data_wide = pd.melt(chart_data.reset_index(), id_vars=["index"])

# Make chart
chart = alt.Chart(chart_data_wide).mark_bar().encode(
          x=alt.X("value", type="quantitative", title="Vote share"),
          y=alt.Y("index", type="nominal", title=""),
          color = alt.condition(alt.datum.index == "Left Candidate", 
                alt.value(left_color),
                alt.value(right_color))
          ).properties(title='Vote Share For Each Candidate')

st.altair_chart(chart, use_container_width=True)

st.markdown("""To see this more clearly, we can look at the area under the curve 
  of the electorate density function for our distribution. Notice how the 
  winning candidate takes more than half of the area under the curve.""")

midpoint = (L + R)/2
df_left = df[df["Position"] <= midpoint]
df_right = df[df["Position"] > midpoint]

# Get chart data
chart_data_wide_left = pd.melt(df_left.reset_index(), 
                  id_vars=["Position"], 
                  value_vars = ["Density"])
chart_data_wide_right = pd.melt(df_right.reset_index(), 
                  id_vars=["Position"], 
                  value_vars = ["Density"])
chart_data_wide_left.rename(columns = {"value":"Density"}, inplace = True)
chart_data_wide_right.rename(columns = {"value":"Density"}, inplace = True)

# Make chart
chart_left= alt.Chart(chart_data_wide_left).mark_area().encode(
    x=alt.X("Position", type = "quantitative"),
    y=alt.Y("Density", type = "quantitative"),
    color =  alt.value(left_color),
    strokeWidth = alt.value(4)
    ).properties(title='Who Wins?')

chart_right= alt.Chart(chart_data_wide_right).mark_area().encode(
    x=alt.X("Position", type = "quantitative"),
    y=alt.Y("Density", type = "quantitative"),
    color = alt.value(right_color),
    strokeWidth = alt.value(4)
    )

point_df = pd.DataFrame([["Left Candidate",L,0.02],["Right Candidate",R,0.02]],
                      columns = [" ","Position","Density"])

chart_position = alt.Chart(point_df).mark_point(filled=True, size = 120).encode(
    x=alt.X("Position", type = "quantitative"),
    y=alt.Y("Density", type = "quantitative"),
    color=alt.Color(" ",
            scale={"range": [LEFT_BLUE_DARK, RIGHT_RED_DARK]}),
    shape = alt.value("triangle"),
    )
# Add scatter plot at left and right candidate positions.

st.altair_chart(chart_left + chart_right + chart_position, use_container_width=True)

q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
        <span class='highlight red'>
          <span class='bold'>
            Discussion Question: <br>
          </span>
        </span>  
        <span class='highlight grey'>
          What are some of the difficulties in making 
           voting mandatory?  Do you know of any countries with mandatory voting?  
           Why do some contries have it while others do not?
        </span>
      </div>
    """

st.markdown(q, unsafe_allow_html=True)

st.subheader("B. Dynamic Candidates")

text_1 = r"If the right candidate has a fixed position, in this case $r$ = "
text_2 = f"{R}, "
text_3 = r"""then it looks like it's always in the best interest of the left 
  candidate to move closer to the right candidate. To see this more clearly, we 
  will compute the vote share as a function of the left candidate position, 
  $\ell$, for a fixed $r$."""

st.markdown(text_1 + text_2 + text_3)

ell_positions = np.linspace(m,R,100)
left_shares = []
right_shares = []
for l in ell_positions:
  left_share = get_left_share(left_position = l, 
                              right_position = R, 
                              model = model)
  left_shares.append(left_share)
  right_share = get_right_share(left_position = l, 
                                right_position = R, 
                                model = model)
  right_shares.append(right_share)

# Get chart data
chart_data = pd.DataFrame()
chart_data["Left"] = ell_positions
chart_data["Left Share"] = left_shares
chart_data["Right Share"] = right_shares
chart_data_wide = pd.melt(chart_data.reset_index(), 
        id_vars=["Left"], 
        value_vars = ["Left Share","Right Share"])
chart_data_wide.rename(columns = {"variable":" ",
                                  "Left":"Candidate Position"}, inplace = True)

# Make chart
chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Candidate Position", 
            type="quantitative", 
            title="Left candidate position"),
    y=alt.Y("value", 
            type = "quantitative", 
            title = "Candidate vote share"),
    color=alt.Color(" ",
            scale={"range": [LEFT_BLUE, RIGHT_RED]}),
    strokeDash=" ",
    strokeWidth = alt.value(4)
    ).properties(
      title='Candidate Vote Share as A Function of Position'
    )


line = alt.Chart(pd.DataFrame({'Vote Share': [0.5]})
                ).mark_rule(strokeDash=[5, 10]).encode(
                    y='Vote Share',
                    color = alt.value(CENTRIST_GREY), 
                    strokeWidth = alt.value(4))

point_df = pd.DataFrame([["Right Candidate",R,0.02]],
                      columns = [" ","Candidate Position","Vote Share"])
point = alt.Chart(point_df).mark_point(filled=True, size = 120).encode(
                    x=alt.X("Candidate Position",
                            type = "quantitative"),
                    y=alt.Y("Vote Share",
                            type = "quantitative", ),
                    color = alt.Color(" ",
                                    scale = {"range": [RIGHT_RED_DARK]}),
                    shape = alt.value("triangle"))

chart = (line + chart + point).resolve_scale(
    color='independent', strokeDash = "independent"
)

st.altair_chart(chart, use_container_width=True)

st.markdown("""As soon as the left candidate crosses the dashed line, they have 
  more than 50% of the votes and therefore they have won the election.""")

q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
        <span class='highlight red'>
          <span class='bold'>
            Discussion Question: <br>
          </span>
        </span>  
        <span class='highlight grey'>
          Where does the left candidate have to be on the 
          spectrum in order to win the election?
        </span>
      </div>
    """

st.markdown(q, unsafe_allow_html=True)

# TODO: show answer - (I dont' think this will work).

# ############## Section III #######################
# st.subheader("III. Opportunistic Candidates")
# ##################################################

st.markdown(r"""From the previous discussion we've seen that candidates might want 
  to change their position on the political spectrum in order to get more votes. 
  Some candidates will do this more eagerly than others. Let's include a measure 
  of __candidate opportism__ into our model, and the larger this measure is, the 
  more eagerly (i.e. more rapidly) a candidate will move towards the 
  opportunistic position.""")

st.markdown(r"""We'll measure the left-wing candidate's opportunism with 
  $\alpha$ and the right-wind candidate's opportunism with $\beta$.  If both 
  candidates are opportunistic, they might for instance follow a system of 
  ordinary differential equations like""")

st.latex(r'''
    \frac{d\ell}{dt}= \alpha \frac{\partial S_L}{\partial \ell}\hspace{1cm} 
    \text{ and }\hspace{1cm}\frac{dr}{dt}= \beta \frac{\partial S_R}{\partial r}
  ''')

st.markdown(r"""Using the sliders below, you can choose your $\alpha$ and 
  $\beta$ values.  Remember, large values mean more opportunism.""")

# Add an alpha slider
alpha = st.slider(r"Choose your $\alpha$ value", 0.0, 2.0, (1.5))

# Add a beta slider
beta = st.slider(r"Choose your $\beta$ value",0.0, 2.0, (1.0))

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

  if y:
    for i in range(10):
        ell_positions.append(y)
        r_positions.append(y)

  return ell_positions, r_positions


ell_positions, r_positions = coalescing_candidates(left_position = L, 
                                                  right_position = R, 
                                                  model = model, 
                                                  alpha = alpha, 
                                                  beta = beta)

# Get chart data
chart_data = pd.DataFrame({"Left Candidate":ell_positions,
                      "Right Candidate":r_positions})
chart_data_wide = pd.melt(chart_data.reset_index(), 
                        id_vars=["index"], 
                        value_vars = ["Left Candidate","Right Candidate"])
chart_data_wide.rename(columns = {"variable":"Candidate"}, inplace = True)

# Make chart
chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("index", type="quantitative", title="Timestep"),
    y=alt.Y("value", type = "quantitative", title = "Candidate position"),
    color=alt.Color('Candidate',scale={"range": [LEFT_BLUE, RIGHT_RED]}),
    strokeDash='Candidate',
    strokeWidth = alt.value(4)
    ).properties(
        title='Candidates Moving Accoring to Steepest Ascent'
    )
st.altair_chart(chart, use_container_width=True)

st.markdown("""When candidates move according to steepest ascent they will 
  eventually meet as some equilibrium point.  In the plot below we show the votes
  share for each candidate at this equilibrium point.""")

left_share = get_left_share(left_position = ell_positions[-1], 
                          right_position = r_positions[-1], 
                          model = model)
right_share = get_right_share(left_position = ell_positions[-1], 
                          right_position = r_positions[-1], 
                          model = model)
left_color, right_color = get_plot_colors(left_share, right_share)

# Get chart data
chart_data = pd.DataFrame([[left_share],[right_share]], 
          columns = ["share"],
          index = ["Left Candidate", "Right Candidate"])
chart_data_wide = pd.melt(chart_data.reset_index(), id_vars=["index"])

# Make chart
chart = alt.Chart(chart_data_wide).mark_bar().encode(
          x=alt.X("value", type="quantitative", title="Vote share at equilibrium"),
          y=alt.Y("index", type="nominal", title=""),
          color = alt.condition(alt.datum.index == "Left Candidate", 
                alt.value(left_color),
                alt.value(right_color))
          ).properties(title='Vote Share For Each Candidate')

st.altair_chart(chart, use_container_width=True)


q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
        <span class='highlight red'>
          <span class='bold'>
            Discussion Question: <br>
          </span>
        </span>  
        <span class='highlight grey'>
          What do you notice?  Is there ever a way for the less eager candidate 
          to win?
        </span>
      </div>
    """

st.markdown(q, unsafe_allow_html=True)

############## Section IV #######################
st.subheader("III. When Voting is Not Mandatory")
#################################################

st.markdown("""In the United States voting is not mandatory. If a voter doesn't 
  feel strongly about either candidate they might choose to stay home. 
  Therefore, it's not always in a candidate's best interest to be overly 
  opportunistic, since they might risk alienating voters on the far ends 
  of their parties.  To account for this, we'll include a measure of __voter 
  loyalty__ into this model.  A high level of voter loyaly means that a voter is 
  likely to stick with a candidate even as their position drifts.""")

st.markdown(r"We'll call this quantity $\gamma$ and define a function ")

st.latex('''
  g(z) = e^{-z/\gamma}.
  ''')

st.markdown("""Then we can express the left and right candidate share of votes 
  as a function of position as """)

st.latex(r'''
  S_L = S_L(\ell, r) = \int_{-\infty}^{\frac{\ell+r}{2}} f(x)\cdot 
  g(\mid \ell - x\mid ) \, \, dx,
  ''')
st.markdown("and")

st.latex(r'''
  S_R = S_R(\ell, r) = \int_{\frac{\ell+r}{2}}^{\infty} f(x)\cdot 
  g(\mid r - x\mid ) \, \, dx.
  ''')

st.markdown(r"""Using the slider below, you can choose your $\gamma$ values. Remember, a greater value of $\gamma$, means the voter is more likely to stick 
  with the candidate, even as their position moves.""")

# Add a gamma slider
gamma = st.slider(r'Choose your $\gamma$ value',0.0, 10.0, (5.0))

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


ell_positions = np.linspace(m,R, 100)
left_shares = []
right_shares = []
for l in ell_positions:
  left_share = left_share_with_g(left_position = l, 
                                right_position = R, 
                                gamma = gamma, 
                                model = model)
  left_shares.append(left_share)

  right_share = right_share_with_g(left_position = l, 
                                  right_position = R, 
                                  gamma = gamma, 
                                  model = model)
  right_shares.append(right_share)

# Prepare chart data
chart_data = pd.DataFrame()
chart_data["Left candidate position"] = ell_positions
chart_data["Left Candidate"] = left_shares
chart_data["Right Candidate"] = right_shares
chart_data_wide = pd.melt(chart_data.reset_index(), 
                          id_vars=["Left candidate position"], 
                          value_vars = ["Left Candidate","Right Candidate"])
chart_data_wide.rename(columns = {"variable":"Candidate"}, inplace = True)

# Make chart
chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Left candidate position", 
            type="quantitative"),
    y=alt.Y("value", 
            type = "quantitative", 
            title = "Candidate vote share"),
    color=alt.Color('Candidate',
            scale={"range": [LEFT_BLUE, RIGHT_RED]}),
    strokeDash='Candidate',
    strokeWidth = alt.value(4)
    ).properties(
        title='Candidate Vote Share as A Function of Position with Voter Loyalty'
    )
st.altair_chart(chart, use_container_width=True)

q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
        <span class='highlight red'>
          <span class='bold'>
            Discussion Question: <br>
          </span>
        </span>  
        <span class='highlight grey'>
          At what point does the left candidate win the election?  Why is it 
          possible for the blue candidate to win with less that 50% of the vote?
        </span>
      </div>
    """

st.markdown(q, unsafe_allow_html=True)

st.markdown("""Next, we will look at how the share of votes changes as a 
  function of candidate position, with the introduction of voter loyalty.  We'll 
  be approximating derivatives """)

st.latex(r'''
  \frac{\partial S_L}{\partial \ell } = \frac{\partial }{\partial \ell}\left[
  \int_{-\infty}^{\frac{\ell + r}{2}} f(x) \cdot 
  g\left(\mid \ell - x\mid \right)\,dx
  \right]
  ''') 

st.markdown("""using the symmetric difference quotient.""")

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


ell_positions = np.linspace(m,R -.001, 100)
left_derivatives = []
for l in ell_positions:
  D = left_derivative(l,right_position = R, gamma = gamma, model = model)
  left_derivatives.append(D)

# Prepare chart data
chart_data = pd.DataFrame()
chart_data["Left candidate position"] = ell_positions
chart_data["Derivatives"] = left_derivatives
chart_data_wide = pd.melt(chart_data, 
                          id_vars=["Left candidate position"], 
                          value_vars = ["Derivatives"])
chart_data_wide.rename(columns = {"value":"Rate of change in vote share"}, 
                        inplace = True)

# Make chart
chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Left candidate position", 
            type="quantitative"),
    y=alt.Y("Rate of change in vote share", 
            type = "quantitative"),
    color=alt.value(LEFT_BLUE),
    strokeWidth = alt.value(4)).properties(
      title='Rate of Change in Left Candidate Vote Share as A Function of Position'
            )


line = alt.Chart(pd.DataFrame({'Rate of change in vote share': [0.], 
                                              "color":["white"]})
                ).mark_rule(strokeDash=[5, 10]).encode(
                    y='Rate of change in vote share',
                    color = alt.value("white"))

st.altair_chart(chart + line, use_container_width=True)

q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
        <span class='highlight red'>
          <span class='bold'>
            Discussion Question: <br>
          </span>
        </span>  
        <span class='highlight grey'>
            What do you notice? Can you find a set of parameters so that the 
            left candidate loses by moving in either direction? Looking at this 
            graph, where are the fixed points and which of these are stable?
        </span>
      </div>
    """

st.markdown(q, unsafe_allow_html=True)


st.header("""III. Discontinuities""")

st.markdown("""The political environment discontinuously impacts the optimal 
  strategy of the candidates.""")

# add l infinite plot.

st.markdown("""You can learn more about the mathematics behind this app in this 
  paper: Börgers, Christoph, Bruce Boghosian, Natasa Dragovic, and Anna Haensch. 
  _A blue sky bifurcation in the dynamics of political candidates._ 
  [arXiv preprint arXiv:2302.07993](https://arxiv.org/abs/2302.07993) (2023).
  """)

st.markdown("""Many thanks to the Tufts Data Intensive Studies Center for 
  supporting this work.
  """)
