import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

# Load pre-trained BERT model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, bert_model

tokenizer, bert_model = load_model_and_tokenizer()

# Function to get sentence embeddings
def get_embeddings(sentence, max_length=128, pooling_strategy="mean"):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    outputs = bert_model(**inputs)

    # Apply the chosen pooling strategy
    if pooling_strategy == "mean":
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    elif pooling_strategy == "max":
        sentence_embedding = outputs.last_hidden_state.max(dim=1).values
    elif pooling_strategy == "cls":
        sentence_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    else:
        raise ValueError("Invalid pooling strategy. Choose from 'mean', 'max', or 'cls'.")

    return sentence_embedding

# Function to calculate cosine similarity
# Optionally allow choice between PyTorch and sklearn implementations
def calculate_similarity(embedding1, embedding2, method="torch"):
    if method == "torch":
        cos_sim = F.cosine_similarity(embedding1, embedding2)
        return cos_sim.item()
    elif method == "sklearn":
        embedding1_np = embedding1.detach().numpy()
        embedding2_np = embedding2.detach().numpy()
        return sklearn_cosine_similarity(embedding1_np, embedding2_np)[0][0]

# Streamlit app UI
st.title(':mag_right: Real-Time Paraphrase Detection')
st.write("Enter two sentences below to check if they are paraphrases.")

sentence1 = st.text_input(":one: Enter the first sentence:")
sentence2 = st.text_input(":two: Enter the second sentence:")

# Add threshold input to the UI
threshold = st.slider("Set similarity threshold:", 0.0, 1.0, 0.8, 0.01)

# Allow users to choose the similarity computation method
method = st.selectbox("Choose similarity computation method:", ["torch", "sklearn"], index=0)

# Allow users to choose the pooling strategy
pooling_strategy = st.selectbox("Choose pooling strategy:", ["mean", "max", "cls"], index=0)

if sentence1 and sentence2:
    # Get embeddings for the input sentences
    embedding1 = get_embeddings(sentence1, pooling_strategy=pooling_strategy)
    embedding2 = get_embeddings(sentence2, pooling_strategy=pooling_strategy)
    
    # Calculate similarity using the selected method
    similarity_score = calculate_similarity(embedding1, embedding2, method=method)

    # Display similarity score
    st.write(f":bar_chart: **Similarity Score:** {similarity_score:.2f}")

    # Check if sentences are paraphrases based on a threshold
    if similarity_score > threshold:
        st.success(":white_check_mark: The sentences are likely paraphrases.")
    else:
        st.error(":x: The sentences are not paraphrases.")
