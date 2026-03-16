import streamlit as st
import torch
import torch.nn as nn
import pickle

# Load TFIDF
with open("tfidf.pkl","rb") as f:
    tfidf = pickle.load(f)

# RNN model
class RNN(nn.Module):
    def __init__(self,input_size):
        super(RNN,self).__init__()
        self.rnn = nn.RNN(input_size,128,batch_first=True)
        self.fc = nn.Linear(128,1)

    def forward(self,x):
        out,_ = self.rnn(x)
        out = out[:,-1,:]
        out = self.fc(out)
        return out

# Load model
input_size = 5000
model = RNN(input_size)
model.load_state_dict(torch.load("sentiment_model.pth", map_location="cpu"))
model.eval()

# Streamlit UI
st.title("🎬 IMDB Sentiment Analysis")
st.write("Enter a movie review")

review = st.text_area("Movie Review")

if st.button("Predict"):
    
    if review.strip()=="":
        st.warning("Please enter a review")
    else:
        review = review.lower()

        vector = tfidf.transform([review]).toarray()
        tensor = torch.tensor(vector).float().unsqueeze(1)

        with torch.no_grad():
            output = model(tensor)
            prediction = torch.sigmoid(output).item()

        if prediction > 0.5:
            st.success("Positive Review 😀")
        else:
            st.error("Negative Review 😡")