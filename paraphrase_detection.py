import streamlit as st
from transformers import BertTokenizer, BertModel, pipeline
import torch
import torch.nn.functional as F

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load a pre-trained summarization model from Hugging Face's Transformers
summarizer = pipeline('summarization')

# Function to get sentence embeddings
def get_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = bert_model(**inputs)
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    cos_sim = F.cosine_similarity(embedding1, embedding2)
    return cos_sim.item()

# Function to get a summary/meaning of the sentence
def get_meaning(sentence):
    summary = summarizer(sentence, max_length=50, min_length=10, do_sample=False)
    return summary[0]['summary_text']

# Streamlit app UI
st.title('Real-Time Paraphrase Detection')
st.write("Enter two sentences below to check if they are paraphrases.")

sentence1 = st.text_input("Enter the first sentence:")
sentence2 = st.text_input("Enter the second sentence:")

if sentence1 and sentence2:
    # Get embeddings for the input sentences
    embedding1 = get_embeddings(sentence1)
    embedding2 = get_embeddings(sentence2)
    
    # Calculate cosine similarity
    similarity_score = cosine_similarity(embedding1, embedding2)

    # Display similarity score
    st.write(f"Cosine Similarity Score: {similarity_score:.2f}")

    # Check if sentences are paraphrases based on a threshold
    threshold = 0.8  # You can adjust this value based on your requirements
    if similarity_score > threshold:
        st.success("The sentences are likely paraphrases.")
    else:
        st.error("The sentences are not paraphrases.")

    # Display the meanings/summaries of the sentences
    st.subheader("Meanings of the Input Sentences")
    st.write("Meaning of the first sentence:")
    st.write(get_meaning(sentence1))

    st.write("Meaning of the second sentence:")
    st.write(get_meaning(sentence2))
