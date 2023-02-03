# Wine_project

## Project Description
We are part of a data science team using the wine data set to develop a machine learning model that will help to predict the quality of wines. 

## Project Goals
- To discover drivers of quality of wine
- Use the drivers to develop a ML program that predicts the quality of wine
- Deliver a report to a data science team

## Questions to answer
- Does a higher content of Free Sulfer Dioxide lead to a higher quality wine?
- Do reds wines have a lower content of Sulfur Dioxide compared to white wines?
- Do higher quality wines have a higher content of citric acid?
- Does higher contents of total suflur dioxider lead to a higher pH level?



## Initial Thoughts and Hypothesis
We believe the main drivers of quality will be chlorides, sulfur dioxide, and acidity. The higher the content of chlorides the higher the quality of the wine. The higher the content of sulfur dioxide the higher the quality of wine. The more acids in the wine the lower the quality of the wine.


## Planning
- Use the aquire.py already used in previous exerices to aquire the data necessary
- Use the code already written prior to aid in sorting and cleaning the data
- Discover main drivers
 - First identify the drivers using statistical analysis
 - Create a pandas dataframe containing all relevant drivers as columns
- Develop a model using ML to determine wine quality based on the top drivers
 - MVP will consist of one model per driver to test out which can most accurately predict quality
 - Post MVP will consist of taking most accurate and running it through multiple models
- Draw and record conclusions


## Data Dictionary
|Target Variable | Definition|
|-----------------|-----------|
| quality | The quality of the wine |

| Feature  | Definition |
|----------|------------|
| fixed acidity |  The total amount of acids of the wine (g/L) |
| volatile acidity |  The amount of organic acids that can be extracted through distillation (g/L) |
| citric acid |  The amount of citric acid in the wine (g/mL) |
| sugar |  The amount of sugar in the wine (g) |
| chlorides | The amount of chloride and sodium in the wine (mg/mL) |
| fso2 | The amount of Free Sulfur Dioxide in the wine (PPM) |
| tso2 | The total amount of Sulfur Dioxide in the wine (PPM) |
| density | g/mL of the wine |
| pH | pH level of the wine |
| alcohol | Alcohol percentage in the wine |
| type | Red or white wine | 


## Recommendations
- Free and Total Sulfur Dioxide are good metrics for determining quality.
- Citric acid had a positive relationship with quality.
- pH did not have a high correlation with quality.