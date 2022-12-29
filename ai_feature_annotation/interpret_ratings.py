from germansentiment import SentimentModel
import pandas as pd

model = SentimentModel()

sent_dict = {'neutral':0,'negative':-1,'positive':1}
texts = [
    "Mit keinem guten Ergebniss","Das ist gar nicht mal so gut",
    "Total awesome!","nicht so schlecht wie erwartet",
    "Der Test verlief positiv.","Sie fährt ein grünes Auto."]

def interprete_review(reviews):       
    results = model.predict_sentiment(reviews)
    results = [sent_dict[result] for result in results]
    return results

review_df = pd.read_pickle('data/poem_reviews.pkl')

for coll in list(review_df.columns[3:]):
    numeric_coll = coll + '_num'
    review_df[numeric_coll] = review_df[coll].apply(lambda x: interprete_review(x))
    rating_coll = coll + '_rating'
    review_df[rating_coll] = review_df[numeric_coll].apply(lambda x: sum(x))

review_df.to_csv('data/poem_reviews_interpreted.csv')
review_df.to_pickle('data/poem_reviews_interpreted.pkl')