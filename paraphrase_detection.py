import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to get sentence embeddings
def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = bert_model(**inputs)
    # Mean pooling to get sentence embeddings
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    cos_sim = F.cosine_similarity(embedding1, embedding2)
    return cos_sim.item()

# Streamlit app UI
st.title('Real-Time Paraphrase Detection')
st.write("Enter two sentences below to check if they are paraphrases.")

sentence1 = st.text_input("Enter the first sentence:")
sentence2 = st.text_input("Enter the second sentence:")

# Slider for dynamic threshold
threshold = st.slider("Set similarity threshold:", 0.5, 1.0, 0.8)

if sentence1 and sentence2:
    # Get embeddings for the input sentences
    embedding1 = get_embeddings(sentence1)
    embedding2 = get_embeddings(sentence2)
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(embedding1, embedding2)

    # Display similarity score
    st.write(f"Cosine Similarity Score: {similarity_score:.2f}")

    # Display a bar chart for visualization
    fig, ax = plt.subplots()
    ax.bar(['Similarity Score'], [similarity_score], color='skyblue')
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score')
    ax.set_title('Cosine Similarity Visualization')
    st.pyplot(fig)

    # Check if sentences are paraphrases based on the threshold
    if similarity_score > threshold:
        st.success("The sentences are likely paraphrases.")
    else:
        st.error("The sentences are not paraphrases.")
