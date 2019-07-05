from sklearn import preprocessing
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
import pandas as pd

def get_review_topic_weights(yelp_data):

    yelp_data['review_length'] = yelp_data.review.map(len)
    yelp_resampled = yelp_data.copy()

    pos_reviews = yelp_data.review[(yelp_data.rating>3)].values
    neg_reviews = yelp_data.review[(yelp_data.rating<=3)].values



    extra_words = ['ve', 'like', 'got', 'Cleveland', 'just',
               'don', 'really', 'said', 'told', 'ok',
               'came', 'went', 'did', 'didn', 'good']

    stop_words = text.ENGLISH_STOP_WORDS.union(extra_words)
    tfidf_pos = TfidfVectorizer(stop_words=stop_words, min_df=10, max_df=0.5,
                        ngram_range=(1,1), token_pattern='[a-z][a-z]+')

    tfidf_neg = TfidfVectorizer(stop_words=stop_words, min_df=10, max_df=0.5,
                        ngram_range=(1,1), token_pattern='[a-z][a-z]+')

    dicty = {'noodles' : 'noodle', 'dishes': 'dish',
         'buns': 'bun', 'asked' : 'ask',
         'pieces' :'piece', 'burgers' : 'burger' ,
        'minutes' : 'minute', 'orders' : 'order', 'waffles' :'waffle'}

    def replace_words(text, dicty):
        for i,j in dicty.items():
            text = text.replace(i,j)
        return text

    neg   = [replace_words(w, dicty) for w in neg_reviews]
    pos   = [replace_words(w, dicty) for w in pos_reviews]


    neg_vectors   = tfidf_neg.fit_transform(neg)
    pos_vectors   = tfidf_pos.fit_transform(pos)
#     total_vectors = tfidf_pos.fit_transform(total)


    num_topics = 10

    nmf_pos = NMF(n_components=num_topics)
    W_pos = nmf_pos.fit_transform(pos_vectors)
    H_pos = nmf_pos.components_

    nmf_neg = NMF(n_components=num_topics)
    W_neg = nmf_neg.fit_transform(neg_vectors)
    H_neg = nmf_neg.components_


    pos_topics = {0:'great offerings', 1:'fresh ingredients', 2:'feel-good food',
                  3:'great pub food ', 4:'great ambiance', 5: 'authentic food',
                  6: 'great value', 7: 'impressive dishes', 8: 'friendly service', 9: 'great location'}

    neg_topics =  {0:'unimpressive offerings', 1:'mediocre traditional food', 2:'mediocre ingredients',
                  3:'difficult reservation', 4:'mediocre pub food', 5: 'unimpressive food',
                  6: 'poor value', 7: 'rude service', 8: 'overrated reputation', 9: 'long wait'}


    df_pos = yelp_data.loc[yelp_data.rating>3]
    df_neg = yelp_data.loc[yelp_data.rating<=3]

    W_pos_df = pd.DataFrame(normalize(W_pos, norm='l1'), columns = list(pos_topics.values()))
    W_pos_df['restaurant'] = df_pos['restaurant']

    W_neg_df = pd.DataFrame(normalize(W_neg, norm='l1'), columns = list(neg_topics.values()) )
    W_neg_df['restaurant'] = df_neg['restaurant']


    new_yelp_data = pd.concat(
                        [df_pos.merge(W_pos_df, how = 'left', on = 'restaurant'),
                         df_neg.merge(W_neg_df, how = 'left', on = 'restaurant')] ,
                        axis = 0
    )

    new_yelp_data = new_yelp_data.fillna(0)

    return new_yelp_data
