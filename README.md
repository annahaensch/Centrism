# ODEs and Mandatory Voting

_This repository exists as an accompaniment to the paper ["ODEs and 
Mandatory Voting" by C. Börgers, N. Dragovic, A. Haensch, 
A. Kirshtein and L. Orr](), where you can find more technical details, 
discussion questions, and suggested homework problems._
            
Mandatory voting has been adopted by over 20 of the world’s 
democracies, including Brazil and Australia, and has been discussed 
in the United States as well.  This repository and its associated Streamlit web app 
contains code implementing tools from ODEs and mathematical modeling to help understand its effects. 
For a population with static beliefs, we explore how candidates 
might adjust their position to maximize their vote share.  We 
also explore how manditory voting might change a candidate's optimal 
strategy.

If you're looking for the interactive web app, head over to [https://centrism.streamlit.app/](https://centrism.streamlit.app/).  

# Using This Repository

If you want to work with the code in this repository, you'll want to start by cloning the repository.  Once that's done, from your terminal you can change into the Centrism directory.

Next, you'll likely want to create a new environment using conda (in case you need it, [installation guide here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)). You will create a new environment (we'll call it centrism_env) with

```
conda create --name centrism_env
```

and activate your new environment, with

```
conda activate centrism_env
```

To run the tools in the libarary will need to install the necessary dependencies. First you'll need to conda install pip and then install the remaining required Python libraries as follows.

```
conda install pip
pip install -U -r requirements.txt
```

Now your environment should be set up to run anything in this library. The Jupyter notebook `getting_started.ipynb` is probably a good place to start. 

If you notice any issues you can feel free to raise an issue directly on the repository, or email anna.haensch@tufts.edu.
