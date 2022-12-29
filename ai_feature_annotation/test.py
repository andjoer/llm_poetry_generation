import pandas as pd

df = pd.read_pickle('data/poem_reviews.pkl')
print(df.head())
print(df.iloc[0]['temp_0.3_prompt1'][3])