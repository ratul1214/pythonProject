# import required sklearn libs
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# import other required libs
import pandas as pd
import numpy as np

# string manipulation libs
import re
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns

categories = [
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'alt.atheism',
 'soc.religion.christian',
]
def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This function cleans the input text by
    - removing links
    - removing special chars
    - removing numbers
    - removing stopwords
    - transforming in lower case
    - removing excessive whitespaces
    Arguments:
        text (str): text to clean
        remove_stopwords (bool): remove stopwords or not
    Returns:
        str: cleaned text
    """
    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove numbers and special chars
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. creates tokens
        tokens = nltk.word_tokenize(text)
        # 2. checks if token is a stopword and removes it
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. joins all tokens again
        text = " ".join(tokens)
    # returns cleaned text
    text = text.lower().strip()
    return text


def get_top_keywords(n_terms):
    """This function returns the keywords for each centroid of the KMeans"""
    df = pd.DataFrame(X.todense()).groupby(clusters).mean()  # groups tf idf vector per cluster
    terms = vectorizer.get_feature_names_out()  # access to tf idf terms
    for i, r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]]))  # for each row of the dataframe, find the n terms that have the highest tf idf score
p = "Located in the heart of Piccadilly in our beautiful flagship store. This central London location, offers so much to explore including restaurants, bars, cultural sites, shopping and more, and only a short walk from the Green Park Tube Station and plenty of bus stops. Excitingly, we are now looking for an experienced and exceptional Floor Manager to lead our busy Tills Department on our Ground Floor. Our Floor Manager will be fully responsible for the tight-knit Tills team, managing their performance through strong leadership skills and will ensure our world class customer service is delivered consistently. As a Floor Manager, you will successfully demonstrate your ability to resolve escalated customers’ requests, queries and complaints. drive strong operational execution across their department. apply sound planning, organisation and time management principles. encourage personal development and coach the team on product knowledge and procedures. demonstrate excellent communication skills, as you will be liaising with various departments such as Buying, Merchandising, HR and Visual Merchandising. Reporting to the Retail Manager, you will be managing a busy team of sales assistants throughout the year, which will increase significantly during the peak Christmas period. Our successful candidate will possess prior experience working in a similar fast-paced and customer-facing role, ideally within the luxury market. You will have proven experience in managing team performance and must be fully flexible and able to work varying shifts, including late store closings, weekends and bank holidays.Reporting to the Retail Manager, you will be managing a busy team of sales assistants throughout the year, which will increase significantly during the peak Christmas period.Our successful candidate will possess prior experience working in a similar fast-paced and customer-facing role, ideally within the luxury market. You will have proven experience in managing team performance and must be fully flexible and able to work varying shifts, including late store closings, weekends and bank holidays.In return, we offer fabulous benefits: A competitive salary. A generous store and restaurant discount of up to 40%.Discretionary annual bonus."

dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, remove=('headers', 'footers', 'quotes'))
documents = nltk.tokenize.sent_tokenize(p)
df = pd.DataFrame(documents, columns=["corpus"])
df['cleaned'] = df['corpus'].apply(lambda x: preprocess_text(x, remove_stopwords=True))


# initialize vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
# fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
# X = vectorizer.fit_transform(df['cleaned'])
X = vectorizer.fit_transform(documents)
# initialize KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
clusters = kmeans.labels_

# initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)
# pass X to the pca
pca_vecs = pca.fit_transform(X.toarray())
# save the two dimensions in x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]

# assign clusters and PCA vectors to columns in the original dataframe
df['cluster'] = clusters
df['x0'] = x0
df['x1'] = x1

cluster_map = {0: "sport", 1: "technology", 2: "religion"} # mapping found through get_top_keywords
df['cluster'] = df['cluster'].map(cluster_map)

# set image size
plt.figure(figsize=(12, 7))
# set title
plt.title("Raggruppamento TF-IDF + KMeans 20newsgroup", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
#  create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="viridis")
plt.show()


