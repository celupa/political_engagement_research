# Methodological and Demographic Predictors of Political Engagement
<br/>

# TLDR
A XGBoost classifier predicting the political engagement of a person has been trained and deployed to AWS via Lambda. The model assesses if a person would benefit from an intervention aiming to educate and re-engage someone in political action.

Test the model by downloading and running "test_aws_lambda.py" (python and "requests" package needed).
<br/>
<br/>

## Context 
What aspects of a person's life influences their political engagement? 

The **World Values Survey** (WVS) is an organization iteratively analyzing people's values on a global scale.
They have been running the experiment for more than 40 years. 
Their questionnaire covers features such as personal values, wellbeing, political perceptions...

This is the first part of 3 series project.
For this first project, we'll attempt to predict the political engagement levels of a person via surface dimensions like sex, education and profession.
The second project will analyse political engagement by adding more complex variables (e.g., political perception).
The final project will analyse the evolution of political engagement over the years. 
<br/>

## Rationale
I believe it's useful to understand what makes a particular person "politically" engaged or not.
Whether we are aware of, adhering or not to this idea, we are political creatures, living in a political world.
Voting and asking for that succulent piece of bread at the table both rely on political undertones.
We are simplifying here, but both of these actions involve the individual's relationship with their past and format their future.  

Wouldn't political apathy result in the relinquishing of our political momentum, thus losing, among other valuable things, our voice to the table?

Understanding political profiles is the first step in a comprehensive process that would help us to:
* **Inform** about the consequences of political engagement 
* **Prevent** political manipulation by identifying vulnerable populations 
* **Support** intervention campaigns to revigorate political engagement
* **Protect** systems and citizens of political decay by keeping track of political public sentiment over time
* ...  
<br/>

## Data
The World Values Survey Data is a questionnaire that covers a wide range of topics.
The data is free to use but is not distributable but can be retrieved via their site or use a custom set I've appended to the project.
You can find the data and other related in-depth documentation here (Wave 7) here: https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp
The 7th wave ran from 2017 to 2022, after which the results were consolidated in one comprehensive dataset.
For this project, only a handful of variables have been retained (see retained_variables.md)
<br/>

## How to
The environment used for analysis is stored in **env.yml**. If you have anaconda installed run the file with the following cmd line: *conda env create -f “path_to_env.yml_file”* (win). This will create a custom conda environemnt with all the dependencies.

Replicate the analyses via **notebook.ipynb**.

Skip the main analyses and run **train.ipynb** to train the model with the final version of the data (wrangled & imputed).

Deploy the app to a local server by first saving the model in train.py with bentoml and running the following cmds (docker required):
* **Build**: bentoml build
* **Containerize**: bentoml containerize MODEL_NAME:TAG
* **Serve**: docker run -it --rm -p 3000:3000 MODEL_NAME:TAG

The model has been deployed on **AWS Lambda**. Try it out via **test_aws_lambda.py**.
<br/>

## Shoutout 
Thank you to the WVS organization for keeping the academic playground accessible and transparent.
<br/>

## Sources
Haerpfer, C., Inglehart, R., Moreno, A., Welzel, C., Kizilova, K., Diez-Medrano J., M. Lagos, P. Norris, E. Ponarin & B. Puranen (eds.). 2022. World Values Survey: Round Seven - Country-Pooled Datafile Version 5.0. Madrid, Spain & Vienna, Austria: JD Systems Institute & WVSA Secretariat. doi:10.14281/18241.20
