from flask import Flask, render_template, request, session
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from nltk.corpus import stopwords
import spacy
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from textacy.extract import keyterms as kt
import re
from more_itertools import split_after
from collections import OrderedDict


#*** Flask configuration#
#os.chdir('c:\\Users\\uname\\desktop\\python')
# Define folder to save uploaded files to process further
UPLOAD_FOLDER = 'C:\\AIML\\capstone\\data'
 
# Define allowed files (for this example I want only csv file)
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__, template_folder='templates', static_folder='data')
# Configure upload file path flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
# Define secret key to enable session
app.secret_key = 'This is your secret key to utilize session in Flask'
 
 
@app.route('/')
def index():
    return render_template('index.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        # upload file flask
        uploaded_df = request.files['uploaded-file']
 
        # Extracting uploaded data file name
        data_filename = uploaded_df.filename
 
        # flask upload file to database (defined uploaded folder in static path)
        uploaded_df.save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
 
        # Storing uploaded file path in flask session
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
 
        return render_template('index_upload_and_show_data_page2.html')
 
@app.route('/DataCleaning',methods=("POST", "GET"))
def DataCleaning():
    # Retrieving uploaded file path from session
    data_file_path = session.get('uploaded_data_file_path', None)
    data_read_file_path=os.path.join(app.config['UPLOAD_FOLDER'], 'datacleaned.csv')
    # read csv file in python flask (reading uploaded csv file from uploaded server location)
    industry_safety_df = pd.read_csv(data_file_path)
    #industry_safety_df.drop("Unnamed: 0", axis=1, inplace=True)
    industry_safety_df.rename(columns={'Data':'Date', 'Countries':'Country', 'Genre':'Gender', 'Employee or Third Party':'Employee type'}, inplace=True)
    industry_safety_df.drop_duplicates(inplace=True)
    #session['industry_safety_df_j']=industry_safety_df.to_json()
    industry_safety_df.to_csv('C:\\AIML\\capstone\\data\datacleaned.csv', index=False)
    #industry_safety_df.to_csv().save(os.path.join(app.config['UPLOAD_FOLDER'], data_filename))
    return render_template('DataCleaning_result.html')

@app.route('/DataPreparation',methods=("POST", "GET"))
def DataPreparation():
    # Retrieving uploaded file path from session
    stop = stopwords.words('english')
    nlp = spacy.load('en_core_web_sm')
    #datacleaned=os.path.join(app.config['UPLOAD_FOLDER'],datacleaned.csv)
    industry_safety_df=pd.read_csv('C:\\AIML\\capstone\\data\datacleaned.csv')
    #industry_safety_df = pd.read_csv(datacleaned)
    industry_safety_df['Cleaned_Description'] = industry_safety_df['Description'].apply(lambda x : x.lower())
    industry_safety_df['Cleaned_Description'] = industry_safety_df['Cleaned_Description'].apply(lambda x: re.sub(' +', ' ', x))
    def remove_punctuations(text):
        return re.sub('\[[^]]*\]', '', text)

    def remove_specialchars(text):
        return re.sub("[^a-zA-Z]"," ",text)

    def remove_stopwords_and_lemmatization(text):
        final_text = []
        text = text.lower()
        text = nltk.word_tokenize(text)
    
        for word in text:
            if word not in set(stopwords.words('english')):
                #lemma = nltk.WordNetLemmatizer()
                #word = lemma.lemmatize(word) 
                final_text.append(word)
        return " ".join(final_text)

    def cleaning(text):
        text = remove_punctuations(text)
        text = remove_specialchars(text)
        text = remove_stopwords_and_lemmatization(text)
        return text

    industry_safety_df['Cleaned_Description'] = industry_safety_df['Cleaned_Description'].apply(cleaning)

    def key_word_extraction(text):
        summaries = []
        y=nlp(text)
        summaries = kt.textrank(y, normalize="lemma", topn=10)
        R = np.array(summaries)[:,0]
        return R

    #industry_safety_df['Summary'] = industry_safety_df['Cleaned_Description'].apply(key_word_extraction)
    industry_safety_df['Summary'] = industry_safety_df['Cleaned_Description'].apply(cleaning)
    
    def tokenize(texts):
        return [nltk.tokenize.word_tokenize(t) for t in texts]

    def key_word_intersection(df):
        summaries = []
        for x in tokenize(df['Cleaned_Description'].to_numpy()):
            keywords = np.concatenate([np.intersect1d(x, ['hand', 'eye', 'finger','fingers','hip','heel' 'ankle', 'frontal region', 'face',
                                                   'foot','leg','neck','hands','eyes','hemiface','legs','feet','eyebrows','eyebrow',
                                                   'forearm','lip','chest','lips','wrist','calf','shoulder','knee','elbow','lumbar area','head','arm'])                              
                                      ])
            dot_sep_sentences = np.array(list(split_after(x, lambda i: i == " ")), dtype=object)
        #dot_sep_sentences='a'
            summary = []
            for i, s in enumerate(dot_sep_sentences):
                summary.append([dot_sep_sentences[i][j] for j, keyword in enumerate(s) if keyword in keywords ])
            summaries.append(','.join([' '.join(x) for x in summary if x]))
        return summaries

#df = pd.DataFrame(text, columns = ['Text'])
    industry_safety_df['Organs'] = key_word_intersection(industry_safety_df)


    def remove_duplicate(string):
        return(','.join(dict.fromkeys(string.split())))

    industry_safety_df['Organs']=industry_safety_df['Organs'].apply(remove_duplicate)
    industry_safety_df.Organs.replace('', "Body")
    industry_safety_df.to_csv('C:\\AIML\\capstone\\data\dataprepared.csv', index=False)
    return render_template('DataPreparation_result.html')

@app.route('/ModelTraining',methods=("POST", "GET"))
def Modeltraining():
    import LSTM_model
    df=pd.read_csv('C:\\AIML\\capstone\\data\datacleaned.csv')
    x_test = df['Gender']
    y_test = df['Accident Level']
    result = LSTM_model.train_LSTM(x_test,y_test)
    return render_template('Modeltraining_result.html')
if __name__=='__main__':
    app.run(debug = True)