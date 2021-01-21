import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
import joblib
from sqlalchemy import create_engine
from collections import OrderedDict

import sys
from nltk.corpus import stopwords
import re



def tokenize(text):
    """Customer tokenizer using nlkt functions.

    Parameters
    ----------
    text : str
        Input text to be tokenized.

    Returns
    -------
    str
        Tokenized text.

    """

    #### Normalize (convert to lower case and remove punctuation) text
    text = re.sub("[^a-z,A-Z,0-9]", " ", text.lower().strip())

    #### Tokenize text to words
    text = word_tokenize(text)

    #### Remove stop words
    text = [i for i in text if i not in stopwords.words('english') ]

    #### Lemmatize
    text = [WordNetLemmatizer().lemmatize(x, pos = 'n') for x in text]
    text = [WordNetLemmatizer().lemmatize(x, pos = 'v') for x in text]

    return text

app = Flask(__name__)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    char_counts = df['message'].apply(lambda x: len(x))
    word_counts = df['message'].apply(lambda x: len(x.split()))

    category_counts = df.drop(columns = ['id', 'message', 'original', 'genre']).sum(axis = 0)


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Histogram(
                    x=char_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Character Counts',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "# of characters in the message"
                }
            }
        },

        {
            'data': [
                Histogram(
                    x=word_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Word Counts in Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "# of words in message",
                    'tickangle': 45,
                    'title_standoff': 45
                }
            }
        },


        {
            'data': [
                Bar(
                    x=category_counts.index,
                    y=category_counts.values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 45,
                    'title_standoff': 1000
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    temp = \
    sorted(zip(df.columns[4:], classification_labels), key = lambda x: (-x[1],x[0]))
    classification_results = OrderedDict()
    for key, val in temp:
        classification_results[key] = val


    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()
