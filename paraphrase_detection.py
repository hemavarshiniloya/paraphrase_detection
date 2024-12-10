import streamlit as st
import spacy
from transformers import BertTokenizer, BertModel, T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn.functional as F

# Load spaCy model
nlp = spacy.load('en_core_web_md')

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

# Function to find similar words using spaCy
def find_similar_words(word, threshold=0.7):
    similar_words = []
    token = nlp.vocab[word]
    if token.has_vector:
        for vocab_word in nlp.vocab:
            if vocab_word.has_vector and vocab_word.is_lower and vocab_word != token:
                similarity = token.similarity(vocab_word)
                if similarity >= threshold:
                    similar_words.append(vocab_word.text)
    return list(set(similar_words))

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
        
        # Find and display similar words for important words in the sentences
        words = sentence1.split()
        for word in words:
            similar_words = find_similar_words(word)
            if similar_words:
                st.write(f"Words similar to '{word}': {', '.join(similar_words[:5])}")  # Display up to 5 similar words
    else:
        st.error("The sentences are not paraphrases.")
