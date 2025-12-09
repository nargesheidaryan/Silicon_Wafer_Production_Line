Wafer Defect Analysis:
This repository contains a short exploratory data analysis (EDA) of a semiconductor manufacturing dataset to investigate whether process or equipment parameters can explain wafer defects. 
The project includes a Python script, data CSV file , and a slide report summarizing the findings.

Contents:
Wafer_Defect_Analysis.py --> main Python analysis script
Wafer_defects_Report.pptx --> presentation with graphs and explanations
README.md --> project documentation

Project goal: 
To identify potential relationships between wafer defect occurrence (particle count and defect labels) and various process parameters such as temperature, gas flow, RF power, pressure, vibration, and tool type.

Methods:
Data cleaning and preprocessing (pandas)
Distribution plots and scatter matrices (matplotlib/seaborn)
Pearson correlation analysis
Comparison of defective vs. non-defective wafers
Basic train/test split for reproducibility

Key observation:
No strong linear correlation exists between any single process parameter and wafer defects.
Parameter ranges were narrow, and the dataset appeared very clean, suggesting low variability.
The lack of significant patterns may also reflect dataset bias (e.g., mostly normal-operation samples, missing contextual variables).
Wafer defects are likely influenced by multivariate interactions or temporal/equipment-specific factors not captured here.

Conclusion:
Simple one-to-one statistical relationships are insufficient to explain wafer defects in this dataset. 
More advanced approaches - such as anomaly detection, clustering, or multivariate machine learning models - would be needed to uncover defect patterns. 
Additional process variables or more diverse data would improve the ability to identify root causes.

Potential next steps:
Apply machine learning classifiers (e.g., random forest, SVM)
Perform time-series or equipment-aging analysis
Investigate tool-specific patterns
Expand dataset with environmental or maintenance data
