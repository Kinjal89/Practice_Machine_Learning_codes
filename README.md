# Practice_Machine_Learning_codes
Here's a machine learning that predicts whether a language is a rich morphology language or a poor morphology language
Language.csv: import pandas as pd\\
from sklearn.tree import DecisionTreeClassifier\\
language= pd.read_csv('Language.csv')\\
language\\
A = language.drop(columns=['morphology'])
b = language['morphology']
model = DecisionTreeClassifier()
model.fit(A, b)
prediction = model.predict([ [1200, 1], [250, 0] ])
prediction

Output: array(['RM', 'PM'], dtype=object)
