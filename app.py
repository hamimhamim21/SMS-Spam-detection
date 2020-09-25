from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
#Loading Naive Bayes model
clf = pickle.load(open(filename, 'rb'))
#count vectorizer
cv=pickle.load(open('tranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = [str(x) for x in request.form.values()][0]
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
		if(my_prediction[0] == 1):
    			output = "ITS A SPAM!"
		elif(my_prediction[0] == 0):
    			output = "NOT A SPAM"
	return render_template('home.html',prediction_text = output)



if __name__ == '__main__':
	app.run(debug=True)