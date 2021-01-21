import sys
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import pandas as pd
import re

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import joblib
from sklearn.pipeline import Pipeline


def load_data(database_filepath):
    """Load data from the database.

    Parameters
    ----------
    database_filepath : str
        Database filepath.

    Returns
    -------
    tuple
        Tuple of features, targets and target names.

    """

    #### Establish connection to database and read the table as pandas df
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table('disaster_response', engine)

    #### define message (X), categories (Y) and category names
    X = df['message']
    df_y = df.drop(columns = ['id', 'message', 'original', 'genre'])
    category_names = df_y.columns.tolist()
    Y = df_y.values

    return X, Y, category_names


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

def build_model():
    """Build the ML model.

    Returns
    -------
    model
        sklearn multioutput classifer model.

    """
    base_model = RandomForestClassifier(n_estimators = 100, max_depth = 200)
    estimator = MultiOutputClassifier(base_model)

    #stop_words = [tokenize(i) for i in stopwords.words('english')]
    pipeline = Pipeline([("tfidf",TfidfVectorizer(tokenizer = tokenize, stop_words = None)), \
                ("estimator", estimator)])
    param_grid = {'estimator__estimator__n_estimators': range(400,499,100), \
                  'estimator__estimator__max_depth': range(400,499,100)}

    model = GridSearchCV(pipeline, param_grid = param_grid, cv = 3, verbose = 5, n_jobs = 1)
    #print(model.get_params().keys())
    return model

def round_list(x, n = 2):
    """Auxiliary function to round elements of a list to n decimal places.

    Parameters
    ----------
    x : list
        List of float values.
    n : int
        Number of decmial places to round the elements of list.

    Returns
    -------
    list
        List with elements rounded to n decimal places.

    """
    try:
        return [round(i,n) for i in x]
    except:
        return x

def evaluate_model(model, X_test, Y_test, category_names):
    """Function to evaluate model predictions on the test set and print classification metrics.

    Parameters
    ----------
    model : sklearn model
        sklearn classification model trained on the training set.
    X_test : np.array
        Test set features.
    Y_test : np.array
        Test set targets.
    category_names : list
        List of category names for each target.

    Returns
    -------
    None

    """
    Y_model = model.predict(X_test)

    try:
        for i,category in enumerate(category_names):
            print("##############")
            print("classification report for category {0}".format(category))
            print(classification_report(Y_test[:,i], Y_model[:,i]))

    except Exception as e:
        print("Failed with exception {0}".format(e))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
