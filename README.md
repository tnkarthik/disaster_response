# Disaster Response Pipeline Project.
### Categorizing disaster messages using Machine Learning for efficient response
<p> The goal of this project is to classify incoming disaster messages to be routed to the relevant agencies. During an active disaster there are many messages that come in and timely routing is very essential for saving lives. This project uses an ETL pipeline using pandas to read and store the messages and a ML pipeline using scikit-learn to classify the disaster messages correctly. This project is part of the Udacity Data Science Nanodegree program. </p>

### Instructions to generate ML model and run the app:

1. To be able to use the app first you need to run the ETL and ML pipelines. The ML model pkl file is not provided in the project's github

2. The ETL pipeline reads the disaster messages from the ./data folder and stores the result into a database.
    - To run ETL pipeline please type the following on the command line from the project root directory

        ```bash
        $ python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        ```

    The above call will create a DisasterResponse.db database in the data folder

3. The ML pipeline creates and trains a ML model on the dataset and stores the model as a pkl file in the models folder.
    - To run ML pipeline please type the following on the command line from the project root directory

        ```bash
        $ python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
        ```

    The above call will save a classifier.pkl model in the models folder which will be used by the flask app

4. The package includes an app created using plotly and flask to characterize new messages
    - To run the app please type the following command in the app's directory

        ```bash
        $ python run.py
        ```

    The app can be accessed via http://127.0.0.1:3001/

5.  Here is the overall structure and description of the files in the Project

    ```bash
    app #### folder with the app files
    | - templates
    | |- master.html #### main page of web app
    | |- go.html #### classification result page of web app
    |- run.py #### Flask file that runs app
    data # Folder with the raw data and the database
    |- disaster_categories.csv #### csv file with the categories for each msg
    |- disaster_messages.csv #### csv file with the msg
    |- process_data.py #### python script to process csv files and save to database
    |- DisasterResponse.db #### database to save clean data to
    models
    |- train_classifier.py #### python code to build and train the ML model
    |- classifier.pkl # saved model
    README.md
    README.html
    docs #### folder with documentation for the functions made using sphinx
    misc #### folder with other miscallaneous python codes to run adhoc analyses
    ```


6. For any feedback and questions please contact me on github.
