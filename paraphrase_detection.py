import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# Load the MRPC dataset from the Hugging Face library
dataset = load_dataset('glue', 'mrpc')

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

# Function to predict if sentences are paraphrases based on a similarity threshold
def predict_paraphrase(sentence1, sentence2, threshold=0.8):
    embedding1 = get_embeddings(sentence1)
    embedding2 = get_embeddings(sentence2)
    similarity_score = cosine_similarity(embedding1, embedding2)
    return 1 if similarity_score > threshold else 0, similarity_score

# Function to evaluate model accuracy on a test dataset
def evaluate_accuracy(test_data, threshold=0.8):
    y_true = []
    y_pred = []

    for item in test_data:
        sentence1 = item['sentence1']
        sentence2 = item['sentence2']
        label = item['label']
        
        prediction, _ = predict_paraphrase(sentence1, sentence2, threshold)
        y_true.append(label)
        y_pred.append(prediction)
    
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

# Use the test split of MRPC for evaluation
test_data = dataset['test']

# Evaluate and print the accuracy
accuracy = evaluate_accuracy(test_data, threshold=0.8)
print(f"Model Accuracy on MRPC Test Set: {accuracy:.2f}")

