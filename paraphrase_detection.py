import streamlit as st
from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# Load pre-trained BERT model and tokenizer for embeddings
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load pre-trained T5 model and tokenizer for paraphrasing
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Function to get sentence embeddings
def get_embeddings(sentence):
    inputs = tokenizer_bert(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = bert_model(**inputs)
    sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    cos_sim = F.cosine_similarity(embedding1, embedding2)
    return cos_sim.item()

# Function to generate paraphrase suggestions
def generate_paraphrases(sentence, num_return_sequences=3):
    # Prepare input for the T5 model
    input_text = f"paraphrase: {sentence}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=128, truncation=True)
    
    # Generate paraphrases
    outputs = t5_model.generate(input_ids, num_return_sequences=num_return_sequences, num_beams=5, early_stopping=True, length_penalty=2.0)
    paraphrases = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrases

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

    # Generate and display paraphrase suggestions for the first sentence
    st.write("Paraphrase Suggestions for the first sentence:")
    paraphrases = generate_paraphrases(sentence1)
    for i, paraphrase in enumerate(paraphrases):
        st.write(f"{i + 1}: {paraphrase}")

