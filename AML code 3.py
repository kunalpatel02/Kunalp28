import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
     

# Dataset
study_hours = np.array([1, 2, 3, 4, 5, 6, 7])
exam_scores = np.array([50, 55, 65, 70, 75, 80, 88])
     

# Put into DataFrame for formula API
df = pd.DataFrame({
    "hours": study_hours,
    "score": exam_scores
})
     

# Fit model using formula API
model = smf.ols("score ~ hours", data=df).fit()
     

print(model.summary())

# ANOVA Table
anova_table = sm.stats.anova_lm(model, typ=2)
     

print(model.summary())
print("\nANOVA Table:\n", anova_table)
