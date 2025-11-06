import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

from tqdm import tqdm

def load_data(corpus_path, min_freq=20):
    corpus_text = ""
    with open(corpus_path, encoding="utf-8") as f:
        corpus_text = f.read()
        
    corpus_text = corpus_text.lower()
    corpus_text = re.sub(r'[^a-z\s]', ' ', corpus_text)
    corpus_text = re.sub(r'\s+', ' ', corpus_text).strip()


    tokens = corpus_text.split()
    word_counts = Counter(tokens)

    vocab = ['<unk>'] + [word for word, count in word_counts.items() if count >= min_freq]
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}, Total tokens: {len(tokens)}")

    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    unk_ix = word_to_ix['<unk>']
    tokens = [word if word in word_to_ix else '<unk>' for word in tokens]
    word_counts = Counter(tokens)

    return tokens, word_counts, word_to_ix, ix_to_word, unk_ix

def create_skipgram_data(tokens, window_size, word_to_ix):
    data = []
    progress_bar = tqdm(tokens, desc="Creating Skip-Gram Data")
    for i, center_word in enumerate(progress_bar):
        if center_word == '<unk>':
            continue
        for j in range(max(0, i - window_size), min(len(tokens), i + window_size + 1)):
            if i == j:
                continue
            context_word = tokens[j]
            if context_word == '<unk>':
                continue
            data.append((word_to_ix[center_word], word_to_ix[context_word]))
    return data

class Word2Vec(nn.Module):
    # TODO: 1. Implement the Word2Vec model with two embedding layers
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.center_embed = nn.Embedding(vocab_size, embedding_dim)
        self.outside_embed = nn.Embedding(vocab_size, embedding_dim)
        self.center_embed.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.outside_embed.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
    
    def forward(self, center_words, context_words, negative_samples):
        v_c = self.center_embed(center_words)
        u_o = self.outside_embed(context_words)
        u_neg = self.outside_embed(negative_samples)
        positive_scores = torch.sum(u_o * v_c, dim=1)
        negative_scores = torch.bmm(v_c.unsqueeze(1), u_neg.transpose(1, 2)).squeeze(1)
        return positive_scores, negative_scores

class NegativeSampler:
    # TODO: 3. Implement negative sampling
    def __init__(self, word_counts, word_to_ix=None, power=0.75):
        words = list(word_counts.keys())
        counts = [word_counts[word] for word in words]
        powered_counts = [count ** power for count in counts]
        self.probs = [p / sum(powered_counts) for p in powered_counts]
        self.word_indices = [word_to_ix[word] for word in words] if word_to_ix else list(range(len(words)))
    
    def get_negative_samples(self, context_word_idx, k):
        sampled_positions = np.random.choice(len(self.word_indices), size=k, replace=True, p=self.probs)
        return [self.word_indices[pos] for pos in sampled_positions]

def train(model, training_data, optimizer, loss_fn, epochs, negative_sampler, k_negative_samples=5):
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data)
        progress_bar = tqdm(training_data, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, (center_word_idx, context_word_idx) in enumerate(progress_bar):
            center_tensor = torch.tensor([center_word_idx], dtype=torch.long)
            context_tensor = torch.tensor([context_word_idx], dtype=torch.long)
            neg_sample_indices = negative_sampler.get_negative_samples(context_word_idx, k_negative_samples)
            neg_tensor = torch.tensor([neg_sample_indices], dtype=torch.long)

            # TODO: 4. Train the model using negative sampling
            optimizer.zero_grad()
            positive_scores, negative_scores = model(center_tensor, context_tensor, neg_tensor)
            loss = loss_fn(positive_scores, negative_scores)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if i % 1000 == 0:
                progress_bar.set_postfix({'loss': f'{total_loss / (i + 1):.4f}'})

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_data):.4f}")

def visualize_embeddings(model, ix_to_word):
    embeddings = model.center_embed.weight.data.cpu().numpy()
    result = PCA(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(12, 10))
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in ix_to_word.items():
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.title("Word2Vec Embeddings with Negative Sampling")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()

if __name__== "__main__":
    print("PyTorch version:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    corpus_path = "corpus.txt"
    tokens, word_counts, word_to_ix, ix_to_word, unk_ix = load_data(f"{corpus_path}", min_freq=1)
    vocab_size = len(word_to_ix)

    training_data = create_skipgram_data(tokens, window_size=5, word_to_ix=word_to_ix)
    print(f"Generated {len(training_data)} pairs.")
    print(f"Vocabulary size: {vocab_size}")

    EMBEDDING_DIM = 32
    LEARNING_RATE = 0.01
    EPOCHS = 5
    K_NEGATIVE_SAMPLES = 5

    model = Word2Vec(vocab_size, EMBEDDING_DIM)
    # TODO: 2. Define the loss function
    def loss_fn(positive_scores, negative_scores):
        positive_loss = -torch.log(torch.sigmoid(positive_scores) + 1e-8)
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_scores) + 1e-8), dim=1)
        return torch.mean(positive_loss + negative_loss)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    negative_sampler = NegativeSampler(word_counts, word_to_ix=word_to_ix)

    train(model, training_data, optimizer, loss_fn, EPOCHS, negative_sampler, K_NEGATIVE_SAMPLES)
    print("Training complete!")

    visualize_embeddings(model, ix_to_word)