
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration
VOCAB = ["A", "C", "G", "U"]
EMBED_DIM = 8

np.random.seed(42)

# Create embedding table
embedding_table = {
    base: np.random.randn(EMBED_DIM)
    for base in VOCAB
}


# 1. Encode nucleotide sequence
def encode_sequence(sequence):

    sequence = sequence.upper()

    if not 5 <= len(sequence) <= 50:
        raise ValueError(
            "Sequence length must be between 5 and 50."
        )

    invalid = set(sequence) - set(VOCAB)

    if invalid:
        raise ValueError(
            f"Invalid nucleotide characters: {invalid}"
        )

    return np.stack([
        embedding_table[base]
        for base in sequence
    ])


# 2. Softmax function
def softmax(x):

    x = x - np.max(
        x,
        axis=-1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


# 3. Self-attention mechanism
def self_attention(X):

    seq_len, embed_dim = X.shape

    # Weight matrices
    W_q = np.random.randn(embed_dim, EMBED_DIM) * 0.1
    W_k = np.random.randn(embed_dim, EMBED_DIM) * 0.1
    W_v = np.random.randn(embed_dim, EMBED_DIM) * 0.1

    # Query, Key and Value
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    # Scaled dot-product attention
    scores = (Q @ K.T) / np.sqrt(EMBED_DIM)

    # Normalize scores
    attention_weights = softmax(scores)

    # Calculate output
    output = attention_weights @ V

    return attention_weights, output


# 4. Plot attention heatmap
def plot_attention(weights, sequence):

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(
        weights,
        cmap="viridis"
    )

    positions = range(len(sequence))

    ax.set_xticks(positions)
    ax.set_yticks(positions)

    ax.set_xticklabels(list(sequence))
    ax.set_yticklabels(list(sequence))

    ax.set_xlabel("Attending TO position")
    ax.set_ylabel("Attending FROM position")

    ax.set_title("mRNA Sequence Self-Attention Weights")

    fig.colorbar(
        image,
        ax=ax,
        label="Attention Weight"
    )

    plt.tight_layout()

    os.makedirs("outputs/figures", exist_ok=True)

    plt.savefig(
        "outputs/figures/attention_heatmap.png",
        dpi=150
    )

    plt.show()


# 5. Main execution
if __name__ == "__main__":

    sequence = "ACGUACGUACGGCUA"

    print("Input mRNA sequence:", sequence)
    print("Sequence length:", len(sequence))

    X = encode_sequence(sequence)

    print("\nEncoded sequence shape:", X.shape)

    weights, output = self_attention(X)

    print("\nAttention matrix shape:", weights.shape)

    print("\nAttention weights:")
    print(weights)

    print("\nAttention output shape:", output.shape)

    # Verify normalization
    assert np.allclose(
        weights.sum(axis=1),
        1.0
    )

    print("\nPASS: Every attention row sums to 1.")

    plot_attention(weights, sequence)

    print("\nAttention heatmap saved successfully!")
    print("Week 2 sequence attention completed.")
np.savetxt(
    "outputs/reports/attention_weights.csv",
    weights,
    delimiter=",",
    fmt="%.6f"
)