# Disaster Response Pipeline Project.
### Categorizing disaster messages using Machine Learning for efficient response
<p> The goal of this project is to classify incoming disaster messages to be routed to the relevant agencies. During an active disaster there are many messages that come in and timely routing is very essential for saving lives. This project uses an ETL pipeline using pandas to read and store the messages and a ML pipeline using scikit-learn to classify the disaster messages correctly. This project is part of the Udacity Data Science Nanodegree program. </p>

### Instructions to generate ML model and run the app:

1. To be able to use the app first you need to run the ETL and ML pipelines. The ML model pkl file is not provided in the project's github

2. The ETL pipeline reads the disaster messages from the ./data folder and stores the result into a database.
    - To run ETL pipeline please type the following on the command line from the project root directory

        ```
        $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```

    The above call will create a DisasterResponse.db database in the data folder

3. The ML pipeline creates and trains a ML model on the dataset and stores the model as a pkl file in the models folder.
    - To run ML pipeline please type the following on the command line from the project root directory

        ```
        $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

    The above call will save a classifier.pkl model in the models folder which will be used by the flask app

4. The package includes an app created using plotly and flask to characterize new messages
    - To run the app please type the following command in the app's directory

        ```
        $ python run.py
        ```

    The app can be accessed via http://127.0.0.1:3001/

For any feedback and questions please contact me on github.
