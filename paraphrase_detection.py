import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Function to get sentence embeddings
def get_embeddings(sentence):
    with torch.no_grad():
        inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
        outputs = bert_model(**inputs)
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)
    return sentence_embedding

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    cos_sim = F.cosine_similarity(embedding1, embedding2)
    return cos_sim.item()

# Adding Font Awesome for icons
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True)

# Custom background styling with animated balloon effect
st.markdown("""
    <style>
        body {
            background-color: #f0f8ff; /* Light blue */
            color: #333;
            font-family: 'Arial', sans-serif;
            overflow: hidden; /* Prevent scrollbars */
        }
        .balloon {
            position: absolute;
            width: 50px;
            height: 70px;
            background-color: #FF6347; /* Balloon color */
            border-radius: 50%;
            animation: floatUp 10s linear infinite;
            opacity: 0.8;
        }
        .balloon:nth-child(1) { left: 10%; animation-delay: 0s; }
        .balloon:nth-child(2) { left: 30%; animation-delay: 2s; }
        .balloon:nth-child(3) { left: 50%; animation-delay: 4s; }
        .balloon:nth-child(4) { left: 70%; animation-delay: 6s; }
        .balloon:nth-child(5) { left: 90%; animation-delay: 8s; }

        @keyframes floatUp {
            0% {
                transform: translateY(0);
            }
            100% {
                transform: translateY(-100vh);
            }
        }
        .stButton>button {
            background-color: #4CAF50; /* Green button */
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #45a049; /* Darker green */
        }
        h1 {
            text-align: center;
            color: #fff; /* White color for text */
            background: linear-gradient(90deg, #ff7eb3, #ff758c); /* Gradient background */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3); /* Shadow effect */
        }
        h3 {
            text-align: center;
            color: #333;
        }
    </style>
    """, unsafe_allow_html=True)

# Add balloon elements to the page
st.markdown("""
    <div class="balloon" style="left: 10%; top: 20%;"></div>
    <div class="balloon" style="left: 30%; top: 40%;"></div>
    <div class="balloon" style="left: 50%; top: 30%;"></div>
    <div class="balloon" style="left: 70%; top: 60%;"></div>
    <div class="balloon" style="left: 90%; top: 80%;"></div>
    """, unsafe_allow_html=True)

# Title with an icon and highlighted style
st.markdown('<h1><i class="fas fa-exchange-alt"></i> Real-Time Paraphrase Detection</h1>', unsafe_allow_html=True)

st.write("Enter two sentences below to check if they are paraphrases.")

# Display icons for sentence inputs with unique keys
st.markdown("<h3><i class='fas fa-comment-alt'></i> Enter the first sentence:</h3>", unsafe_allow_html=True)
sentence1 = st.text_input("Sentence 1", key="sentence1")

st.markdown("<h3><i class='fas fa-comment-alt'></i> Enter the second sentence:</h3>", unsafe_allow_html=True)
sentence2 = st.text_input("Sentence 2", key="sentence2")

# Display a button and an icon separately
if st.button("Check Similarity"):
    st.markdown("<i class='fas fa-search'></i>", unsafe_allow_html=True)
    if sentence1 and sentence2:
        with st.spinner("Calculating similarity..."):
            embedding1 = get_embeddings(sentence1)
            embedding2 = get_embeddings(sentence2)
            similarity_score = cosine_similarity(embedding1, embedding2)

        # Display similarity score
        st.write(f"Cosine Similarity Score: {similarity_score:.2f}")

        # Check if sentences are paraphrases based on a threshold
        threshold = 0.8
        if similarity_score > threshold:
            st.markdown(
                "<h3 style='color: green;'>"
                "<i class='fas fa-check-circle'></i> The sentences are likely paraphrases! 😊"
                "</h3>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<h3 style='color: red;'>"
                "<i class='fas fa-times-circle'></i> The sentences are not paraphrases. 😕"
                "</h3>", unsafe_allow_html=True)
