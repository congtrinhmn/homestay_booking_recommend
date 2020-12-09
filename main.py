import ast
import json

from flask import Flask, jsonify

from collaborative_filtering import recommend

app = Flask(__name__)


def get_list_of_dict(keys, list_of_tuples):
    list_of_dict = [dict(zip(keys, values)) for values in list_of_tuples]
    return list_of_dict


@app.route('/')
def index():
    return "<h1>Welcome to Homestay Booking Recommend System</h1>"


@app.route('/api/recommend/<int:uid>')
def hello_world(uid):
    keys = ("place_id", "rating")
    v = get_list_of_dict(keys, recommend(uid))
    print(len(v))
    response = app.response_class(
        response=json.dumps({"body": v, "length": len(v)}),
        status=200,
        mimetype='application/json'
    )
    return response


if __name__ == '__main__':
    app.run(port=4455)
