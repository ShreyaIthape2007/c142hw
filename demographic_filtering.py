import pandas as pd
import numpy as np

df1 = pd.read_csv('articles.csv')


df1 = df1.sort_values('total_events',ascending = False)

output = df1[["url", "title", "text", "lang", "total_events"]].head(20).values.tolist()