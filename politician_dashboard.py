""" Political Centrism Dashboard 

This code launches a streamlit app 

"""
import streamlit as st
import altair as alt

from load_css import local_css
from functions import *


local_css("style.css")


st.title("ODEs and Mandatory Voting")

st.markdown("""Mandatory voting has been adopted by over 20 of the world’s 
            democracies, including Brazil and Australia, and has been discussed 
            in the United States as well.  In this web app we present some tools
            from ODE's and mathematical modeling to help understand its effects. 
            For a population with static beliefs, we'll explore how candidates 
            might adjust their position to maximize their vote share.  We'll 
            also explore how manditory voting might change a candidate's optimal 
            strategy.""")

st.markdown("""This web app is exists as an accompaniment to the paper [_ODEs and 
            Mandatory Voting_ by C. Börgers, N. Dragovic, A. Haensch, 
            A. Kirshtein and L. Orr](), where you can find more technical details 
            along with our answers to the dicussion questions posed below and 
            some suggested homework problems.
            """)

st.markdown("""Let's get started.""")

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
          <span class='bold'>
            Discussion Question: <br>
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
option = st.selectbox(
        '''How many modes would you like your population to have?  
        This is the _M_ in the equations above.  A set of viable means,
        variances, and weights will be provided to you below, but you can
        edit them if you wish.''',
        ('1', '2', '3','4','5'),
        index = 1)

means_dict = {1: "0.0",
              2: "-1.0, 1.0",
              3: "-2.0, 0.0, 2.0",
              4: "-3.0, -1.0, 1.0, 3.0",
              5: "-4.0, -2.0, 0.0, 2.0, 4.0"
              }

# Set defaults values for means, weights, and variances
means_value = means_dict[int(option)]
variances_value = ", ".join([str(0.5) for i in range(int(option))])
weights_value = ", ".join([str(np.around(1/int(option), decimals = 2)
                                ) for i in range(int(option))])

# Get means, weights, and variances as text input
c1, c2, c3 = st.columns(3)
means = c1.text_input(label = "means", value = means_value)
variances = c2.text_input(label = "variances", value = variances_value)
weights = c3.text_input(label = "weights", value = weights_value)

# Cast means, weights, and variances to floats
means = np.array([float(m.strip(" ")) for m in means.split(",")])
variances = np.array([float(s.strip(" ")) for s in variances.split(",")])
weights = np.array([float(w.strip(" ")) for w in weights.split(",")])
weights = weights/ np.sum(weights)

msg = f"You must select {option} each of means, weights and variances."
assert len(means) == len(variances), msg
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

# Display chart
st.altair_chart(chart, use_container_width=True)


############## Section II #######################
st.header("II. When Voting is Mandatory")
#################################################

st.subheader("A. Static Candidates")
st.markdown(r"""Let's simulate a simple election in which everybody votes.  Suppose 
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

# Get left and right candidate position.
L, R = st.slider("",float(m), float(M), (-1.,2.), step = 0.25)

# Compute left and right candidate population share
left_share = get_left_share(df = df, left_position = L, right_position = R, model = model)
right_share = get_right_share(df = df, left_position = L, right_position = R, model = model)
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

# Display chart
st.altair_chart(chart, use_container_width=True)

st.markdown("""To see this more clearly, we can look at the area under the curve 
  of the electorate density function for our distribution. Notice how the 
  winning candidate takes more than half of the area under the curve.""")

# Compute midpoint and assign voter ranges to candidates
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

# Display chart
chart_right= alt.Chart(chart_data_wide_right).mark_area().encode(
    x=alt.X("Position", type = "quantitative"),
    y=alt.Y("Density", type = "quantitative"),
    color = alt.value(right_color),
    strokeWidth = alt.value(4)
    )

# Add candidate markers to plot
point_df = pd.DataFrame([["Left Candidate",L,0.02],["Right Candidate",R,0.02]],
                      columns = [" ","Position","Density"])

chart_point = alt.Chart(point_df).mark_point(filled=True, size = 120).encode(
    x=alt.X("Position", type = "quantitative"),
    y=alt.Y("Density", type = "quantitative"),
    color=alt.Color(" ",
            scale={"range": [LEFT_BLUE_DARK, RIGHT_RED_DARK]}),
    shape = alt.value("triangle"),
    )

st.altair_chart(chart_left + chart_right + chart_point, use_container_width=True)

st.markdown("""
  If voting is mandatory, then we assume everyone in the population shows up and votes 
  for the candidate whose position is closest to their beliefs.
  """)
# TODO: add some text about mandatory voting.

q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
          <span class='bold'>
            Discussion Questions: <br>
          </span>
        <span class='highlight grey'>
          <ul>
          <li> How is the border between the left candidate votes and right 
          candidate votes determined and how would you express it in terms of 
          the candidate positions?
          <li>What are some of the difficulties in making 
           voting mandatory?  Do you know of any countries with mandatory voting?  
           Why do some countries have it while others do not?
          </ul>
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

# Compute vote shares with changing left candidate position
ell_positions = np.linspace(m,R,100)
left_shares = []
right_shares = []
for l in ell_positions:
  left_share = get_left_share(df = df,
                              left_position = l, 
                              right_position = R, 
                              model = model)
  left_shares.append(left_share)
  right_share = get_right_share(df = df,
                                left_position = l, 
                                right_position = R, 
                                model = model)
  right_shares.append(right_share)

# Get chart data
chart_data = pd.DataFrame()
chart_data["Left Position"] = ell_positions
chart_data["Left"] = left_shares
chart_data["Right"] = right_shares
chart_data_wide = pd.melt(chart_data.reset_index(), 
        id_vars=["Left Position"], 
        value_vars = ["Left","Right"])
chart_data_wide.rename(columns = {"variable":" ",
                                  "Left Position":"Candidate Position",
                                  "value": "Proportion"}, inplace = True)

# Make chart
chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Candidate Position", 
            type="quantitative", 
            title="Left candidate position"),
    y=alt.Y("Proportion", 
            type = "quantitative", 
            title = "Proportion of population"),
    color=alt.Color(" ",
            scale={"range": [LEFT_BLUE, RIGHT_RED]}),
    strokeDash=" ",
    strokeWidth = alt.value(4),
    tooltip=[alt.Tooltip("Candidate Position:Q", format=",.2f"),
            alt.Tooltip("Proportion:Q", format=",.2f"),
            alt.Tooltip(" :N")]
    ).properties(
      title="""Proportion of Population Voting for Each Candidate as a Function of Left Candidate Position"""
    )


# Add dashed line at 0.5
line = alt.Chart(pd.DataFrame({'Vote Share': [0.5]})
                ).mark_rule(strokeDash=[5, 10]).encode(
                    y='Vote Share',
                    color = alt.value('black'), 
                    strokeWidth = alt.value(2),
                    tooltip=alt.value(None))

# Display chart
st.altair_chart(line + chart, use_container_width=True)

st.markdown("""As soon as the left candidate crosses the dashed line, they have 
  more than 50% of the votes and therefore they have won the election.""")

q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
          <span class='bold'>
            Discussion Questions: <br>
          </span>
        <span class='highlight grey'>
        <ul>
          <li>Why does proportion of the population voting for the left candidate 
          get so close to 1, but never quite reach it? 
          <li>Do the proportions always sum to 1? Why or why not?
          <li>Where does the left candidate have to be on the 
          spectrum in order to win the election?
        </ul>
        </span>
      </div>
    """

st.markdown(q, unsafe_allow_html=True)

st.markdown(r"""From the previous discussion we've seen that candidates might want 
  to change their position on the political spectrum in order to get more votes. 
  Some candidates will do this more eagerly than others. Let's include a measure 
  of __candidate opportunism__ into our model, and the larger this measure is, the 
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

# Get alpha value from slider
alpha = st.slider(r"Choose your $\alpha$ value", 0.0, 2.0, (1.0))

# Get beta value from slider
beta = st.slider(r"Choose your $\beta$ value",0.0, 2.0, (0.2))

# Compute left and right candidate positions as candidates move
ell_positions, r_positions = coalescing_candidates(left_position = L, 
                                                  right_position = R, 
                                                  model = model, 
                                                  alpha = alpha, 
                                                  beta = beta)

# Get chart data
chart_data = pd.DataFrame({"Left":ell_positions,
                      "Right":r_positions})
chart_data_wide = pd.melt(chart_data.reset_index(), 
                        id_vars=["index"], 
                        value_vars = ["Left","Right"])
chart_data_wide.rename(columns = {"variable":" "}, inplace = True)

# Make chart
chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("index", type="quantitative", title="Timestep"),
    y=alt.Y("value", type = "quantitative", title = "Candidate position"),
    color=alt.Color(" ",scale={"range": [LEFT_BLUE, RIGHT_RED]}),
    strokeDash=" ",
    strokeWidth = alt.value(4),
    tooltip=[alt.Tooltip("index:Q", format=",.2f"),
            alt.Tooltip("value:Q", format=",.2f"),
            alt.Tooltip(" :N")]
    ).properties(
        title='Candidates Moving According to Steepest Ascent'
    )

# Display chart
st.altair_chart(chart, use_container_width=True)

st.markdown("""When candidates move according to steepest ascent they will 
  eventually meet as some collision point.  In the plot below we show the votes
  share for each candidate at this point.""")

# Get proportion voting for each candidate at time of collision
left_share = get_left_share(df = df, 
                          left_position = ell_positions[-1], 
                          right_position = r_positions[-1], 
                          model = model)
right_share = get_right_share(df = df,
                          left_position = ell_positions[-1], 
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
          x=alt.X("value", type="quantitative", title="Vote share at collision"),
          y=alt.Y("index", type="nominal", title=""),
          color = alt.condition(alt.datum.index == "Left Candidate", 
                alt.value(left_color),
                alt.value(right_color))
          ).properties(title='Proportion of Population Voting For Each Candidate')

# Display chart
st.altair_chart(chart, use_container_width=True)


q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
          <span class='bold'>
            Discussion Questions: <br>
          </span> 
        <span class='highlight grey'>
        <ul>
          <li>What happens to the equilibrium position when the left candidate 
          is very opportunistic and right candidate is not?
          <li>Is there ever a way for the less eager candidate 
          to win?
          <li> This model proposed here doesn't allow the candidates to cross 
          over each other.  What would happen if we loosened these restrictions?  
          Would our formulas still work? 
        </ul>
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
  loyalty__ into this model.  A high level of voter loyalty means that a voter is 
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

st.markdown(r"""Using the slider below, you can choose your $\gamma$ values. 
  Remember, a greater value of $\gamma$, means the voter is more likely to stick 
  with the candidate, even as their position moves.""")

# Get gamma value from slider
gamma = st.slider(r'Choose your $\gamma$ value',0.0, 5.0, (3.0))


# Compute proportion voting for each candidate with voter loyalty
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
chart_data["Left"] = left_shares
chart_data["Right"] = right_shares
chart_data_wide = pd.melt(chart_data.reset_index(), 
                          id_vars=["Left candidate position"], 
                          value_vars = ["Left","Right"])
chart_data_wide.rename(columns = {"variable":" "}, inplace = True)

# Make chart
chart = alt.Chart(chart_data_wide).mark_line().encode(
    x=alt.X("Left candidate position", 
            type="quantitative"),
    y=alt.Y("value", 
            type = "quantitative", 
            title = "Proportion of population"),
    color=alt.Color(' ',
            scale={"range": [LEFT_BLUE, RIGHT_RED]}),
    strokeDash=' ',
    strokeWidth = alt.value(4)
    ).properties(
        title='Proportion of Population Voting for Each Candidate as A Function of Position with Voter Loyalty'
    )

# Display chart
st.altair_chart(chart, use_container_width=True)

q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
          <span class='bold'>
            Discussion Questions: <br>
          </span> 
        <span class='highlight grey'>
        <ul>
          <li>Why does proportion of the population voting for the left candidate 
          get so close to 1, but never quite reach it? 
          <li>Do the proportions always sum to 1? Why or why not?
          <li>Where does the left candidate have to be on the 
          spectrum in order to win the election?
          <li>At what point does the left candidate win the election?  Why is it 
          possible for the blue candidate to win with less that 50% of the vote?
        </ul>
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

# Compute change in vote share as a funciton of loyalty

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
    strokeWidth = alt.value(4),
    tooltip=[alt.Tooltip("Left candidate position:Q", format=",.2f"),
            alt.Tooltip("Rate of change in vote share:Q", format=",.2f")
            ]).properties(
      title='Rate of Change in Left Candidate Vote Share as A Function of Position'
            )

# Add line at 0.
line = alt.Chart(pd.DataFrame({'Rate of change in vote share': [0.], 
                                              "color":["white"]})
                ).mark_rule(strokeDash=[5, 10]).encode(
                    y='Rate of change in vote share',
                    color = alt.value("black"),
                    strokeWidth = alt.value(2),
                    tooltip=alt.value(None))

# Display chart
st.altair_chart(line + chart, use_container_width=True)

q = """
      <div style="margin-bottom: 10px; margin-left: 5px">
          <span class='bold'>
            Discussion Questions: <br>
          </span>
        <span class='highlight grey'>
          <ul>
            <li>What do you notice? Can you find a set of parameters so that the 
            left candidate loses by moving in either direction? Looking at this 
            graph, where are the fixed points and which of these are stable?
          </ul>
        </span>
      </div>
    """

st.markdown(q, unsafe_allow_html=True)


st.header("""III. Discontinuities""")

st.markdown(r"""The political environment discontinuously impacts the optimal 
  strategy of the candidates.  In the chart below we example the final position 
  of the left-hand candidate as a function of $\gamma$ value.  To save computational
  costs, this is a static graph, and we assume that $\ell = -1$, $r = 2$, $\alpha = 1$ 
  and $\beta = 0.2$.  We're assuming a static underlying bimodal population with 
  equally weighted centers at -1 and 1 with variance 0.5.""")

# Read in data for left-candidate final position.
df = pd.read_csv("discontinuity.csv", index_col = 0)
chart_data_wide = pd.melt(df, 
                          id_vars=["gamma"], 
                          value_vars = ["final_left_position"])
chart_data_wide.rename(columns = {"value":"Final left candidate position",
                                  "gamma":"Gamma"}, 
                        inplace = True)

# Make chart
chart = alt.Chart(chart_data_wide).mark_point().encode(
    x = alt.X("Gamma:Q", scale = alt.Scale(domain=(2.5, 3.5))),
    y="Final left candidate position:Q",
    color=alt.value(LEFT_BLUE),
    strokeWidth = alt.value(4),
    tooltip=[alt.Tooltip("Final left candidate position:Q", format=",.2f"),
            alt.Tooltip("Gamma:Q", format=",.2f")]
    ).properties(
      title="Left Candidate's Final Position as a Function of Gamma" 
            )

# Display chart
st.altair_chart(chart, use_container_width=True)

st.markdown("""If you want to try to generate this graph with different values, 
  you can check out the accompanying script `optimal_left_position.py` in the 
  Github repository.""")

st.markdown("""You can learn more about the nature of these discontinuities in 
  _A blue sky bifurcation in the dynamics of political candidates_ by C. 
  Börgers, B. Boghosian, N. Dragovic, and A. Haensch,  
  [arXiv preprint arXiv:2302.07993](https://arxiv.org/abs/2302.07993) (2023).
  """)

st.markdown("""All of the code used to create this web app is written in Python
  and is available on Github at [https://github.com/annahaensch/Centrism]
  (https://github.com/annahaensch/Centrism).
  """)


st.markdown("""Many thanks to the Tufts Data Intensive Studies Center for 
  supporting this work.
  """)
