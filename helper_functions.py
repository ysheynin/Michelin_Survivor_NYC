import numpy as np
from scipy.stats import linregress, kurtosis, skew
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from sklearn import preprocessing
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize




def get_review_topic_weights(yelp_data):

    yelp_data['review'] = (
                        yelp_data.review.str
                       .replace('\\\\xc2', '')
                       .str.replace('\\\\xa0', '')
                        .str.replace('\\\xa0', '')
                       .str.lower()
                       .str.replace('\d+', '')
                       .str.replace(r'[^\w\s]+', '')
                        .str.replace('cocktails', 'cocktail')
                        .str.replace('zzs', '')
                        .str.replace('xc', '')
                        .str.replace('xa', '')
                        .str.replace('zz', '')
                        .str.replace('       ', '')
                        .str.replace('eellent', 'excellent')
                        .str.replace(r'\bthe\b', '')
                        .str.replace(r'\band\b', '')
                        .str.replace(r'\bas\b', '')
                        .str.replace(r'\bof\b', '') )

    # yelp_data.date   =  yelp_data.date.str.replace('Updatedreview', '')                  )
    yelp_data.date   = pd.to_datetime(yelp_data.date.str.replace('Updatedreview', ''))
    yelp_data        = yelp_data.drop_duplicates(keep = False)

    pos_reviews = yelp_data.review[(yelp_data.rating>3)].values
    neg_reviews = yelp_data.review[(yelp_data.rating<=3)].str.replace('\\\xa0', '').values


    extra_words = ['ve', 'like', 'got', 'just',
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

    num_topics = 10

    nmf_pos = NMF(n_components=num_topics)
    W_pos = nmf_pos.fit_transform(pos_vectors)
    H_pos = nmf_pos.components_

    nmf_neg = NMF(n_components=num_topics)
    W_neg = nmf_neg.fit_transform(neg_vectors)
    H_neg = nmf_neg.components_


    pos_topics = {0:'great offerings', 1:'fresh ingredients', 2:'feel-good food',
                      3:'great pub food', 4:'great ambiance', 5: 'authentic food',
                      6: 'great value', 7: 'impressive dishes', 8: 'friendly service', 9: 'great location'}

    neg_topics =  {0:'unimpressive offerings', 1:'mediocre traditional food', 2:'mediocre ingredients',
                      3:'difficult reservation', 4:'mediocre pub food', 5: 'unimpressive food',
                      6: 'poor value', 7: 'rude service', 8: 'overrated reputation', 9: 'long wait'}

    df_pos = yelp_data.loc[yelp_data.rating>3]
    df_neg = yelp_data.loc[yelp_data.rating<=3]

    new_topics = ['menu', 'food_quality', 'service', 'value']


    W_pos_df = pd.DataFrame(normalize(W_pos, norm='l1'), columns = list(pos_topics.values()))

    W_pos_df['menu']         =   W_pos_df['great offerings']
    W_pos_df['food_quality'] = (W_pos_df['feel-good food'] + W_pos_df['great pub food']
                                + W_pos_df['authentic food'] +  W_pos_df['impressive dishes'])
    W_pos_df['service']      = W_pos_df['friendly service'] + W_pos_df['great ambiance']
    W_pos_df['value']        = W_pos_df['great location'] + W_pos_df['great value']
    df_pos = pd.concat([df_pos.reset_index(drop = True), W_pos_df[new_topics]], axis = 1)



    W_neg_df = pd.DataFrame(normalize(W_neg, norm='l1'), columns = list(neg_topics.values()) )
    W_neg_df['menu']         =   -1 * W_neg_df['unimpressive offerings']
    W_neg_df['food_quality'] =   -1 * (W_neg_df['mediocre traditional food'] + W_neg_df['mediocre pub food']
                                + W_neg_df['unimpressive food'] +  W_neg_df['mediocre ingredients'])

    W_neg_df['service']      = -1 * (W_neg_df['difficult reservation'] + W_neg_df['rude service'] + W_neg_df['long wait'])
    W_neg_df['value']        = -1 * (W_neg_df['poor value'] + W_neg_df['overrated reputation'])
    df_neg = pd.concat([df_neg.reset_index(drop = True), W_neg_df[new_topics]], axis = 1)

    def threshold(number):
        if abs(number) > .25:
            return(int(1)) * np.sign(number)
        else:
            return(int(0))

    for each in new_topics:
        df_neg[each] =  df_neg[each].apply(threshold).astype(int)
        df_pos[each] =  df_pos[each].apply(threshold).astype(int)

    new_yelp_data = pd.concat(
                            [df_pos,
                             df_neg] ,
                            axis = 0
        ).fillna(0)


    return new_yelp_data



def get_subset_dict(predict_year, michelin_data, yelp_data):

    evaluation_cutoff = 100 # in days
    year_data = dict()

    next_year = 'stars_{}'.format(str(predict_year))
    this_year = 'stars_{}'.format(str(predict_year-1))
    michelin_subset  = michelin_data.loc[michelin_data[this_year] != 0]
    michelin_subset  = michelin_subset.set_index('name', drop = True)
    michelin_subset  = michelin_subset.loc[list(set(michelin_subset.index) & set(yelp_data.restaurant.unique()))]

    try:
        target = pd.Series((michelin_subset[next_year] - michelin_subset[this_year]) < 0, index = michelin_subset.index).astype(int)
    except:
        target = []

    def check_if_subset(restaurant):
        return restaurant in michelin_subset.index

    yelp_subset_index = yelp_data.restaurant.apply(check_if_subset)
    subset_data  = yelp_data.loc[yelp_subset_index]

    subset_data  = subset_data.loc[subset_data.date < dt.datetime(predict_year, 1, 1)
                                   -  dt.timedelta(days = evaluation_cutoff)]
    year_data['michelin'] = michelin_subset
    year_data['target']   = target
    year_data['yelp']     = subset_data

    return year_data

def trend(series):
    try:
        slope                     = linregress(range(0,len(series)), series)[0]
    except:
        slope = 0
    return slope

def count_char(series):
    num_char = 0
    for each in series:
        num_char += len(each)
    try:
        count =  num_char /len(series)
    except:
        count = 0
    return count

def make_div_features(restaurant_df, predict_year):

    rating_cols = ['slope', 'mean', 'kurtosis', 'skew', 'median', 'std',
                'var', 'number_reviews', 'avg_length_reviews']

    review_cols = ['food_quality', 'menu', 'service', 'value']

    div_df   = pd.DataFrame(columns = rating_cols + review_cols)
    total_df = div_df.copy()

    # make relative time series
    restaurant_df.index          = pd.date_range(start = '2000-01-01', end =  '2000-12-31', periods=restaurant_df.shape[0])
    #resample four times for different measures
    div_df['mean']               = restaurant_df.rating.resample('4M').mean().reset_index(drop = True)
    div_df['slope']              = restaurant_df.rating.resample('4M').apply(trend).reset_index(drop = True)
    div_df['kurtosis']           = restaurant_df.rating.resample('4M').apply(kurtosis).reset_index(drop = True)
    div_df['std']                = restaurant_df.rating.resample('4M').apply(np.std).reset_index(drop = True)
    div_df['var']                = restaurant_df.rating.resample('4M').apply(np.std).reset_index(drop = True) ** 2
    div_df['median']             = restaurant_df.rating.resample('4M').apply(np.median).reset_index(drop = True)
    div_df['skew']               = restaurant_df.rating.resample('4M').apply(skew).reset_index(drop = True)
    div_df['number_reviews']     = restaurant_df.rating.resample('4M').apply(len).reset_index(drop = True)
    div_df['avg_length_reviews'] = restaurant_df.review.resample('4M').apply(count_char).reset_index(drop = True)



    total_df['mean']               = [restaurant_df.rating.mean(), restaurant_df.rating.mean()]
    total_df['slope']              = [trend(restaurant_df.rating),trend(restaurant_df.rating)]
    total_df['kurtosis']           = [restaurant_df.rating.kurtosis(), restaurant_df.rating.kurtosis()]
    total_df['std']                = [np.std(restaurant_df.rating), np.std(restaurant_df.rating)]
    total_df['var']                = [np.std(restaurant_df.rating) ** 2, np.std(restaurant_df.rating) ** 2]
    total_df['median']             = [np.median(restaurant_df.rating), np.median(restaurant_df.rating)]
    total_df['skew']               = [skew(restaurant_df.rating), skew(restaurant_df.rating)]
    total_df['number_reviews']     = [len(restaurant_df.rating), len(restaurant_df.rating)]
    total_df['avg_length_reviews'] = [np.sum(restaurant_df.review.apply(len))/restaurant_df.shape[0], np.sum(restaurant_df.review.apply(len))/restaurant_df.shape[0]
                                     ]
    total_df['first_review']       = ((dt.datetime(predict_year, 1, 1)  -  dt.timedelta(days = 100)) - min(restaurant_df.date)).days

    for each in review_cols:
        div_df[each]   = restaurant_df[each].resample('4M').mean().reset_index(drop = True)
        total_df[each] = restaurant_df[each].mean()

    ratings = dict()
    ratings['divs']  = div_df
    ratings['total'] = total_df

    return ratings

def process_restaurant_yelp_data(restaurant, data_df, predict_year):
    yelp_df          = data_df['yelp']
    michelin_df      = data_df['michelin']
    resto_df         = yelp_df.loc[yelp_df.restaurant == restaurant]
    feature_dict     = make_div_features(resto_df, predict_year)


    return feature_dict


def get_features_and_predicted(data_df, predict_year):

    columns = ['name', 'first_review', 'n_stars']
    measures = ['slope', 'mean', 'kurtosis', 'skew', 'median', 'std',
                'var', 'number_reviews', 'avg_length_reviews', 'food_quality', 'menu', 'service', 'value']

    for measure in measures:
        for div in range(0,5):
            if div < 4:
                columns.append('div{}_{}'.format(div+1, measure))
            else:
                columns.append('total_{}'.format(measure))


    feature_set        = pd.DataFrame(columns = columns, index = data_df['yelp'].restaurant.unique())


    ndivs = 5
    for resto in feature_set.index:

        rating_features = process_restaurant_yelp_data(resto, data_df, predict_year)


        row = dict()
        row['name'] = resto
        row['first_review'] = rating_features['total']['first_review'][0]
        row['n_stars']      = data_df['michelin'].loc[resto]['stars_{}'.format(str(predict_year-1))]
        for measure in measures:
            for div in range(0,ndivs):
                if div+1 < ndivs:
                    row['div{}_{}'.format(div+1, measure)] = rating_features['divs'][measure][div]

                else:
                    row['total_{}'.format(measure)] = rating_features['total'][measure].loc[0]

                feature_set.loc[resto] = row

    feature_set = feature_set.fillna(feature_set.mean(axis = 0))
    target      = pd.DataFrame(data_df['target'], columns = ['target'])
    target['name'] = target.index
    target      = target.reset_index(drop = True)
    full_data   = feature_set.merge(target, how = 'left', on = 'name')
    full_data   = full_data.set_index('name', drop = True)

    y           = full_data.target
    X           = full_data.drop(columns = 'target')
    X           = X.fillna(X.mean())
    X           = scale_X(X)
    X           = pd.get_dummies(X, columns = ['n_stars'], drop_first= True, prefix = 'michelin_')

    return X , y

def scale_X(X):
    scaler  = preprocessing.MinMaxScaler(feature_range=(0, 1))
#     X = X.set_index('name', drop = True)
    scaler.fit(X)
    return pd.DataFrame(scaler.transform(X), columns = X.columns, index = X.index).fillna(0)


# over sample minority


def make_resampled(X,y):
 #input DataFrame
 #X →Independent Variable in DataFrame\
 #y →dependent Variable in Pandas DataFrame format
    sm = SMOTE(random_state = 0)
    X_new, y_new = sm.fit_sample(X, y)

    return  pd.DataFrame(X_new, columns = X.columns), pd.Series(y_new)

def get_train_and_holdout(predict_year, michelin_data, yelp_data):
    train                      = {'X' : pd.DataFrame(), 'y': pd.Series()}
    holdout                    = dict()

    predict_year_dict          = get_subset_dict(predict_year, michelin_data, yelp_data)
    holdout['X'], holdout['y'] = get_features_and_predicted(predict_year_dict, predict_year)


    for year in range(2008, predict_year):
        year_dict = dict()
        year_dict    =  get_subset_dict(year, michelin_data, yelp_data)#.drop('food_quality'4))
        year_dict['X'], year_dict['y']   = get_features_and_predicted(year_dict, year)
        train['X']                =  pd.concat([train['X'], year_dict['X']], axis = 0)
        train['y']                =  pd.concat([train['y'], year_dict['y']], axis = 0)
    return train, holdout


def run_model(train, holdout, cols, clf = LogisticRegression(), thresh = .5, plot_on = True):

    clf.fit(train['X'][cols], train['y'])
    predicted = clf.predict(holdout['X'][cols])
    probs = clf.predict_proba(holdout['X'][cols])
    probs = probs[:,1]
    probs = np.where(probs > thresh, 1, 0)

    print('Mean Accuracy: %0.2f' % metrics.accuracy_score(holdout['y'], predicted) )
    print('Classification Report: \n', metrics.classification_report(holdout['y'], predicted))
    print('Confusion Matrix: \n', metrics.confusion_matrix(holdout['y'], predicted))
    print(metrics.accuracy_score(holdout['y'], predicted))

    fpr, tpr, threshold = metrics.roc_curve(holdout['y'], probs)

    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc.png', dpi = 300)
    plt.show()

    return clf


def get_rfecv_features(train, clf = LogisticRegression()):
    rfecv = RFECV(estimator=clf, step=1, cv=10,
                  scoring='roc_auc', min_features_to_select = 10)

    rfecv.fit(train['X'], train['y'])
    selected_cols = train['X'].columns[rfecv.support_]

    print("Optimal number of features :{}".format(rfecv.n_features_) )

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    return selected_cols
