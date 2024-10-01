import torch
from GPT4 import GPT4

def main():
    # Model hyperparameters
    vocab_size = 10000
    embed_size = 512
    num_layers = 6
    heads = 8
    ff_hidden_size = 2048
    max_length = 100
    dropout = 0.1

    # Create GPT-4 model (simplified version)
    model = GPT4(vocab_size, embed_size, num_layers, heads, ff_hidden_size, max_length, dropout)

    # Sample input
    sample_input = torch.randint(0, vocab_size, (1, max_length))  # Batch size of 1 and max_length tokens
    mask = None  # No masking in this simple example

    # Forward pass
    output = model(sample_input, mask)
    print(output)

if __name__ == "__main__":
    main()
