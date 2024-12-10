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

# Add custom CSS and JavaScript for animations
st.markdown("""
    <style>
        /* Full-screen background with floating numbers and alphabets */
        body {
            overflow: hidden;
            position: relative;
            font-family: Arial, sans-serif;
        }
        .floating {
            position: absolute;
            color: rgba(255, 255, 255, 0.5);
            font-size: 2rem;
            animation: float 15s linear infinite;
        }
        .floating-number {
            animation-duration: 20s;
        }
        .floating-alphabet {
            animation-duration: 25s;
        }
        @keyframes float {
            0% {
                top: 0;
                left: 0;
                transform: translate(0, 0);
            }
            100% {
                top: 100%;
                left: 100%;
                transform: translate(-100%, -100%);
            }
        }
        /* Style for content */
        .content {
            position: relative;
            z-index: 10;
            color: #000;
        }
    </style>
    <script>
        // Function to create floating numbers and alphabets
        function createFloatingElements() {
            const container = document.body;

            // Generate numbers
            for (let i = 0; i < 10; i++) {
                let number = document.createElement('div');
                number.className = 'floating floating-number';
                number.textContent = Math.floor(Math.random() * 10);
                number.style.top = Math.random() * 100 + 'vh';
                number.style.left = Math.random() * 100 + 'vw';
                number.style.opacity = Math.random() * 0.5 + 0.3;
                container.appendChild(number);
            }

            // Generate alphabets
            const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
            for (let i = 0; i < 10; i++) {
                let letter = document.createElement('div');
                letter.className = 'floating floating-alphabet';
                letter.textContent = alphabet.charAt(Math.floor(Math.random() * alphabet.length));
                letter.style.top = Math.random() * 100 + 'vh';
                letter.style.left = Math.random() * 100 + 'vw';
                letter.style.opacity = Math.random() * 0.5 + 0.3;
                container.appendChild(letter);
            }
        }
        window.onload = createFloatingElements;
    </script>
    """, unsafe_allow_html=True)

# Title with an icon
st.markdown('<h1 style="text-align: center;"><i class="fas fa-exchange-alt"></i> Real-Time Paraphrase Detection</h1>', unsafe_allow_html=True)

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
                "<h3 style='color: green; text-align: center;'>"
                "<i class='fas fa-check-circle'></i> The sentences are likely paraphrases! 😊"
                "</h3>", unsafe_allow_html=True)
        else:
            st.markdown(
                "<h3 style='color: red; text-align: center;'>"
                "<i class='fas fa-times-circle'></i> The sentences are not paraphrases. 😕"
                "</h3>", unsafe_allow_html=True)
