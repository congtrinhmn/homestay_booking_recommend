import json
import csv
import pandas as pd

with open('/Users/mba0022/dataset/dataset_01_ha_noi_2020-11-16.json') as f:
    places1: list = json.load(f)
with open('/Users/mba0022/dataset/dataset_38_da_nang_2020-12-02.json') as f:
    places2: list = json.load(f)
with open('/Users/mba0022/dataset/dataset_79_ho_chi_minh_2020-11-16.json') as f:
    places3: list = json.load(f)

with open('data_30k.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['listing_id', 'reviewer_id', 'rating'])

    places = places2
    print(len(places))
    for place in filter(lambda x: len(list(x['reviews'])) > 0, places):
        for review in place['reviews']:
            writer.writerow([review['listingId'], review['author']['id'], review['rating']])
