import csv

import ast
import json

from flask import Flask, jsonify, request

from recommender import train

app = Flask(__name__)


def get_list_of_dict(keys, list_of_tuples):
    list_of_dict = [dict(zip(keys, values)) for values in list_of_tuples]
    return list_of_dict


@app.route('/')
def index():
    return "<h1>Welcome to Homestay Booking Recommend System</h1>"


@app.route('/api/train')
def training():
    train()
    return jsonify(is_success=True)


@app.route('/api/recommend/<int:uid>')
def recommend(uid):
    print(uid)
    with open('output_30k.csv', "r") as readFile:
        reader = list(csv.reader(readFile))
        a = list(filter(lambda x: x[0] == str(uid), reader))
        b = []
        if len(a) != 0:
            b = a[0][1]
            print(b)
            print(type(b))

    readFile.close()
    return jsonify(recommend=b)


@app.route('/api/add_review', methods=['POST'])
def add_review():
    place_id = request.form['place_id']
    user_id = request.form['user_id']
    rating = request.form['rating']

    with open('data_30k.csv', "r") as readFile:
        reader = list(csv.reader(readFile))
        reader.insert(1, (place_id, user_id, rating))

    with open('data_30k.csv', "w") as writeFile:
        writer = csv.writer(writeFile)
        for line in reader:
            writer.writerow(line)

    readFile.close()
    writeFile.close()
    return jsonify(is_success=True)


@app.route('/api/get_review/<int:uid>')
def get_review_of_user(uid):
    with open('data_30k.csv', "r") as readFile:
        reader = list(csv.reader(readFile))
        a = list(filter(lambda x: x[1] == str(uid), reader))

    print(a)
    readFile.close()
    return jsonify(review_of_user=a)


if __name__ == '__main__':
    app.run(port=4455)
