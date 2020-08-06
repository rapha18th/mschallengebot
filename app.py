from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

df = pd.read_csv("dummy.csv")
df.dropna(inplace=True)

vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((df.Question, df.Answer)))

Question_vectors = vectorizer.transform(df.Question)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/get")
def get_bot_response():
    while True:
        # Read user input
        userText = request.args.get('msg')

        # Locate the closest question
        input_question_vector = vectorizer.transform([userText])

        # Compute similarities
        similarities = cosine_similarity(input_question_vector, Question_vectors)

        # Find the closest question
        closest = np.argmax(similarities, axis=1)

        # Print the correct answer
        return str(df.Answer.iloc[closest].values[0])
  
         
if __name__ == '__main__':
    port = int(os.getenv('PORT'))
    print("Starting app on port %d" % port)
    app.run(debug=False, port=port, host='0.0.0.0')

# running the app on the local machine on port 5000
'''if __name__ == "__main__":
    app.run(port=5000, debug=True)'''