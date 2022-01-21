import pandas as pd
import numpy as np

df1 = pd.read_csv('articles.csv')

df1['title'] = df1['title'].str.lower()

from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words = 'english')
count_matrix = count.fit_transform(df1["title"])

from sklearn.metrics.pairwise import cosine_similarity
cs = cosine_similarity(count_matrix, count_matrix)

df1 = df1.reset_index()
indices = pd.Series(df1.index, index = df1['contentId'])

def get_recommendations(title, cosine_sim):
  ind = indices[title]
  sim = list(enumerate(cosine_sim[ind]))
  sim = sorted(sim, key=lambda x: x[1], reverse= True)
  sim = sim[1:11]
  articles = [i[0] for i in sim]
  return df1[["url", "title", "text", "lang", "total_events"]].iloc[articles].values.tolist()



