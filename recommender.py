from collections import defaultdict

import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import GridSearchCV
from surprise.model_selection import KFold


def train():
    df = pd.read_csv('data_200k.csv')

    df.info()

    df.nunique()

    # Check count of reviews given by user

    df.reviewer_id.value_counts()

    # Check count of reviews every listings got

    df.listing_id.value_counts()

    df.rating.value_counts()

    # Removing entries with rating 0.

    df = df[df.rating != 0]

    df.shape

    # As We are going to use collaborative filtering algorithm to predict ratings, we will limit data by considering
    # listings which got at least 10 reviews and users wo have rated more than 3 listings. if there are m users and n
    # listings, matrix size will be m*n which is very big.

    count_places = df['listing_id'].value_counts()
    print('count_places')
    print(count_places)
    df = df[df['listing_id'].isin(count_places[count_places >= 10].index)]

    df.shape

    count_reviewers = df['reviewer_id'].value_counts()
    print('count_reviewers')
    print(count_reviewers)
    df = df[df['reviewer_id'].isin(count_reviewers[count_reviewers > 3].index)]

    df.describe()

    df.head()

    df1 = df[['reviewer_id', 'listing_id', 'rating']]
    df1.head()
    print('df1.nunique()')
    print(df1.nunique())

    # A reader is still needed but only the rating_scale param is requiered.
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df1[['reviewer_id', 'listing_id', 'rating']], reader)

    data1 = data

    raw_ratings = data.raw_ratings

    # shuffle ratings if need
    # random.shuffle(raw_ratings)

    # A = 90% of the data, B = 10% of the data
    threshold = int(.9 * len(raw_ratings))
    A_raw_ratings = raw_ratings[:threshold]
    B_raw_ratings = raw_ratings[threshold:]

    data.raw_ratings = A_raw_ratings  # data is now the set A

    # Select best algorithm with grid search.
    print('Grid Search...')

    param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005, 0.1],
                  'reg_all': [0.4, 0.6], 'n_factors': [100, 500]}
    grid_search = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    grid_search.fit(data)
    print('best RMSE score')
    print(grid_search.best_score['rmse'])

    print('combination of parameters that gave the best RMSE score')
    print(grid_search.best_params['rmse'])
    algorithm = grid_search.best_estimator['rmse']

    # retrain on the whole set A
    train_set = data.build_full_trainset()
    algorithm.fit(train_set)

    # Compute biased accuracy on A
    test_set = train_set.build_testset()
    predictions = algorithm.test(test_set)
    print('Biased accuracy on A,', end='   ')
    accuracy.rmse(predictions, verbose=True)
    print('len(predictions)')
    print(len(predictions))

    # Compute unbiased accuracy on B
    test_set = data.construct_testset(B_raw_ratings)  # testset is now the set B
    predictions = algorithm.test(test_set)
    print('Unbiased accuracy on B,', end=' ')
    accuracy.rmse(predictions)
    print('len(predictions)')
    print(len(predictions))

    # define a cross-validation iterator
    kf = KFold(n_splits=7)
    # algo = SVD(n_factors=500, n_epochs=5, lr_all=0.1)

    for train_set, test_set in kf.split(data1):
        # train and test algorithm.
        algorithm.fit(train_set)

        predictions = algorithm.test(test_set)

        # Compute and print Root Mean Squared Error
        accuracy.rmse(predictions, verbose=True)

    # Build anti test set for predicting top 3 ratings for user

    # retrain on the whole set A
    train_set = data.build_full_trainset()
    algorithm.fit(train_set)

    test_set = train_set.build_anti_testset()
    predictionsAll = algorithm.test(test_set)
    print('Accuracy on whole data set,', end='   ')
    accuracy.rmse(predictionsAll, verbose=True)
    print('len(predictions)')
    print(len(predictionsAll))

    def get_top_n_recommendations(predictions, topN=3):
        top_recs = defaultdict(list)
        for uid, iid, true_r, est, _ in predictions:
            top_recs[uid].append((iid, est))

        for uid, user_ratings in top_recs.items():
            user_ratings.sort(key=lambda x: x[1], reverse=True)
            top_recs[uid] = user_ratings[:topN]
        return top_recs

    print('getting top 20 recommendations')

    top_20_recommendations = get_top_n_recommendations(predictionsAll, 20)

    print('Total predictions calculated are %d' % len(predictionsAll))

    # Writing top 3 recommendations calculated to csv file

    dfo = pd.DataFrame(columns=['UserId', 'Recommended Listing,Rating'])
    i = 0
    for uid, user_ratings in top_20_recommendations.items():
        # print(uid, top3_recommendations[uid])
        row = [uid, top_20_recommendations[uid]]
        dfo.loc[i] = row
        i = i + 1
    dfo.to_csv('output_200k.csv', index=False)

    print("Wrote recommendations for each user in csv file")


train()
