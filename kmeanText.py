import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from nltk import tokenize
nltk.download('punkt')
p = "Located in the heart of Piccadilly in our beautiful flagship store. This central London location, offers so much to explore including restaurants, bars, cultural sites, shopping and more, and only a short walk from the Green Park Tube Station and plenty of bus stops. Excitingly, we are now looking for an experienced and exceptional Floor Manager to lead our busy Tills Department on our Ground Floor. Our Floor Manager will be fully responsible for the tight-knit Tills team, managing their performance through strong leadership skills and will ensure our world class customer service is delivered consistently. As a Floor Manager, you will successfully demonstrate your ability to resolve escalated customers’ requests, queries and complaints. drive strong operational execution across their department. apply sound planning, organisation and time management principles. encourage personal development and coach the team on product knowledge and procedures. demonstrate excellent communication skills, as you will be liaising with various departments such as Buying, Merchandising, HR and Visual Merchandising. Reporting to the Retail Manager, you will be managing a busy team of sales assistants throughout the year, which will increase significantly during the peak Christmas period. Our successful candidate will possess prior experience working in a similar fast-paced and customer-facing role, ideally within the luxury market. You will have proven experience in managing team performance and must be fully flexible and able to work varying shifts, including late store closings, weekends and bank holidays.Reporting to the Retail Manager, you will be managing a busy team of sales assistants throughout the year, which will increase significantly during the peak Christmas period.Our successful candidate will possess prior experience working in a similar fast-paced and customer-facing role, ideally within the luxury market. You will have proven experience in managing team performance and must be fully flexible and able to work varying shifts, including late store closings, weekends and bank holidays.In return, we offer fabulous benefits: A competitive salary. A generous store and restaurant discount of up to 40%.Discretionary annual bonus."

documents = nltk.tokenize.sent_tokenize(p)
print(documents)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 3
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cat is hungry."])
prediction = model.predict(Y)
print(prediction)