{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "8f847d00-0b21-4de9-a610-18aef2cfe2b8",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2026-03-15 21:13:59.954 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.234 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\vinod\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2026-03-15 21:14:00.235 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.237 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.238 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.239 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.240 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.241 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.243 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.244 Session state does not function when running a script without `streamlit run`\n",
      "2026-03-15 21:14:00.246 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.247 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.248 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.250 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.250 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.252 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2026-03-15 21:14:00.253 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import pickle\n",
    "\n",
    "# Load TFIDF\n",
    "with open(\"tfidf.pkl\",\"rb\") as f:\n",
    "    tfidf = pickle.load(f)\n",
    "\n",
    "# RNN model\n",
    "class RNN(nn.Module):\n",
    "    def __init__(self,input_size):\n",
    "        super(RNN,self).__init__()\n",
    "        self.rnn = nn.RNN(input_size,128,batch_first=True)\n",
    "        self.fc = nn.Linear(128,1)\n",
    "\n",
    "    def forward(self,x):\n",
    "        out,_ = self.rnn(x)\n",
    "        out = out[:,-1,:]\n",
    "        out = self.fc(out)\n",
    "        return out\n",
    "\n",
    "# Load model\n",
    "input_size = 5000\n",
    "model = RNN(input_size)\n",
    "model.load_state_dict(torch.load(\"sentiment_model.pth\", map_location=\"cpu\"))\n",
    "model.eval()\n",
    "\n",
    "# Streamlit UI\n",
    "st.title(\"🎬 IMDB Sentiment Analysis\")\n",
    "st.write(\"Enter a movie review\")\n",
    "\n",
    "review = st.text_area(\"Movie Review\")\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    \n",
    "    if review.strip()==\"\":\n",
    "        st.warning(\"Please enter a review\")\n",
    "    else:\n",
    "        review = review.lower()\n",
    "\n",
    "        vector = tfidf.transform([review]).toarray()\n",
    "        tensor = torch.tensor(vector).float().unsqueeze(1)\n",
    "\n",
    "        with torch.no_grad():\n",
    "            output = model(tensor)\n",
    "            prediction = torch.sigmoid(output).item()\n",
    "\n",
    "        if prediction > 0.5:\n",
    "            st.success(\"Positive Review 😀\")\n",
    "        else:\n",
    "            st.error(\"Negative Review 😡\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4a2e7088-de8c-43d4-957f-a05db0d5128c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
