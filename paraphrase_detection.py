import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

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
st.title(':mag_right: Real-Time Paraphrase Detection')
st.write("Enter two sentences below to check if they are paraphrases.")

sentence1 = st.text_input(":one: Enter the first sentence:")
sentence2 = st.text_input(":two: Enter the second sentence:")

if sentence1 and sentence2:
    # Get embeddings for the input sentences
    embedding1 = get_embeddings(sentence1)
    embedding2 = get_embeddings(sentence2)
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(embedding1, embedding2)

    # Display similarity score
    st.write(f":bar_chart: **Cosine Similarity Score:** {similarity_score:.2f}")

    # Check if sentences are paraphrases based on a threshold
    threshold = 0.8  # You can adjust this value based on your requirements
    if similarity_score > threshold:
        st.success(":white_check_mark: The sentences are likely paraphrases.")
    else:
        st.error(":x: The sentences are not paraphrases.")
