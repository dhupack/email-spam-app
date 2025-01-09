import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title("SMS Spam Detection Model")
st.write("*This is a Machine Learning application to Predict SMS as spam or ham made by Deepak Kumar*")
    

input_sms = st.text_input("Enter the SMS for Prediction")

if st.button('Predict'):
    if input_sms:
    # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tk.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
         st.write("Please type Email to Predict")       

# import streamlit as st
# import pickle

# # Load the trained model and vectorizer
# model = pickle.load(open("model.pkl", 'rb'))
# cv = pickle.load(open("vectorizer.pkl", 'rb'))

# # Set the title and description
# st.title("SMS Spam Detection Model")
# st.write("*This is a Machine Learning application to Predict SMS as spam or ham made by Deepak Kumar*")

# # Text area for user input
# user_input = st.text_area("Enter the SMS for Prediction", height=150)

# # Predict button
# if st.button("Predict"):
#     if user_input.strip():  # Ensure input is not empty
#         # Preprocess and predict
#         data = [user_input]
#         vectorized_data = cv.transform(data).toarray()
#         result = model.predict(vectorized_data)

#         # Display the prediction
#         if result[0] == 0:
#             st.success("The SMS is not spam")
#         else:
#             st.error("The SMS is spam")
#     else:
#         st.warning("Please type an SMS to predict.")
     