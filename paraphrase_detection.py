import streamlit as st
from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn.functional as F
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Ensure that the NLTK data is downloaded for WordNet
nltk.download('wordnet')
nltk.download('punkt')

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load T5 model and tokenizer for paraphrase generation
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')

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

# Function to generate a paraphrase using T5
def generate_paraphrase(sentence):
    input_text = f"paraphrase: {sentence}"
    input_ids = t5_tokenizer.encode(input_text, return_tensors='pt')
    output = t5_model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
    paraphrase = t5_tokenizer.decode(output[0], skip_special_tokens=True)
    return paraphrase

# Function to find synonyms using WordNet
def find_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

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
        
        # Generate and display a paraphrase for the first sentence
        paraphrase = generate_paraphrase(sentence1)
        st.write(f"Suggested Paraphrase: {paraphrase}")
        
        # Find and display synonyms for important words in the sentences
        words = word_tokenize(sentence1)
        synonyms = {word: find_synonyms(word) for word in words if wordnet.synsets(word)}
        st.write("Synonyms for important words in the sentence:")
        for word, syns in synonyms.items():
            if syns:
                st.write(f"{word}: {', '.join(syns[:5])}")  # Display up to 5 synonyms per word
    else:
        st.error("The sentences are not paraphrases.")
