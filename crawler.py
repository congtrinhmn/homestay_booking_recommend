import json
import csv
import pandas as pd
#
# with open('/Users/mba0022/Downloads/dataset_my-task_2020-11-16_08-26-24-635.json') as f:
#     places: list = json.load(f)
#
# with open('/Users/mba0022/Downloads/data-airbnb.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(["room_id", "user_id", "rating"])
#
#     for place in filter(lambda x: len(list(x['reviews'])) > 10, places):
#         for review in place['reviews']:
#             writer.writerow([review['listingId'], review['author']['id'], review['rating']])
#
# # Parse JSON into an object with attributes corresponding to dict keys.
# print(len(places))
# print(len(list(filter(lambda x: len(list(x['reviews'])) > 0, places))))

r_cols = ['room_id', 'user_id', 'rating']

ratings_base = pd.read_csv('/Users/mba0022/Downloads/data-airbnb.csv', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
print(len(rate_train))
