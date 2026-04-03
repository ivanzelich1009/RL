import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    random_split
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

N = 32
SEQ_LEN = N * N

BATCH_SIZE = 64

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

LR = 3e-4
EPOCHS = 20

NUM_TOTAL_SAMPLES = 50000

TRAIN_FRAC = 0.8

BURN_IN = 5000
DECORRELATION_STEPS = 25

beta_c = 0.5 * np.log(1 + np.sqrt(2))

print("=" * 60)
print("DEVICE:", DEVICE)
print("=" * 60)


#Wolff cluster update algorithm for 2D Ising model

def wolff_update(spins, beta):

    N = spins.shape[0]

    p_add = 1 - np.exp(-2 * beta)

    x = np.random.randint(N)
    y = np.random.randint(N)

    cluster_spin = spins[x, y]

    cluster = set([(x, y)])
    stack = [(x, y)]

    while stack:

        i, j = stack.pop()

        neighbors = [
            ((i + 1) % N, j),
            ((i - 1) % N, j),
            (i, (j + 1) % N),
            (i, (j - 1) % N),
        ]

        for ni, nj in neighbors:

            if (ni, nj) not in cluster:

                if spins[ni, nj] == cluster_spin:

                    if np.random.rand() < p_add:

                        cluster.add((ni, nj))
                        stack.append((ni, nj))

    for i, j in cluster:
        spins[i, j] *= -1

    return spins


# Dataset generation using Wolff algorithm to sample from the critical 2D Ising model distribution

def generate_ising_dataset(num_samples):

    print("\n" + "=" * 60)
    print("GENERATING ISING DATASET")
    print("=" * 60)

    spins = np.random.choice([-1, 1], size=(N, N))

    samples = []

    print("\nBurn-in phase...")

    for _ in tqdm(range(BURN_IN)):
        spins = wolff_update(spins, beta_c)

    print("\nCollecting samples...")

    for _ in tqdm(range(num_samples)):

        for _ in range(DECORRELATION_STEPS):
            spins = wolff_update(spins, beta_c)

        samples.append(spins.copy())

    samples = np.array(samples)

    print("\nDataset shape:", samples.shape)

    return samples

class IsingDataset(Dataset):

    def __init__(self, samples):

        samples = samples.reshape(samples.shape[0], -1)

        # -1 -> 0
        # +1 -> 1

        samples = ((samples + 1) // 2).astype(np.int64)

        self.samples = torch.tensor(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x = self.samples[idx]
        
        BOS_TOKEN = 2 # Special token for beginning of sequence

        bos = torch.full((1,), BOS_TOKEN, dtype=torch.long)

        inp = torch.cat([bos, x[:-1]], dim=0)

        target = x

        return inp, target


'''Positional Encoding for Transformer

        We build Fourier modes on the discrete torus:
        cos(2π kx x / N)
        sin(2π kx x / N)
        cos(2π ky y / N)
        sin(2π ky y / N) 
        for multiple frequencies k.
        This allows the model to learn spatial patterns and correlations in the 2D lattice.'''

class FourierPositionalEncoding2D(nn.Module):

    def __init__(self, d_model, N, num_frequencies=16):

        super().__init__()

        self.N = N
        self.d_model = d_model

        pe = torch.zeros(N, N, d_model)

        frequencies = torch.arange(
            1,
            num_frequencies + 1
        )

        for x in range(N):
            for y in range(N):

                features = []

                for k in frequencies:

                    features.append(
                        math.cos(2 * math.pi * k * x / N)
                    )

                    features.append(
                        math.sin(2 * math.pi * k * x / N)
                    )

                    features.append(
                        math.cos(2 * math.pi * k * y / N)
                    )

                    features.append(
                        math.sin(2 * math.pi * k * y / N)
                    )

                features = torch.tensor(features)

                if len(features) < d_model:

                    padding = torch.zeros(
                        d_model - len(features)
                    )

                    features = torch.cat([
                        features,
                        padding
                    ])

                else:

                    features = features[:d_model]

                pe[x, y] = features

        # Flatten:

        pe = pe.reshape(N * N, d_model)

        self.register_buffer(
            "pe",
            pe.unsqueeze(0)
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# Transformer-based autoregressive model for 2D Ising configurations

class IsingTransformer(nn.Module):

    def __init__(
        self,
        N,
        embed_dim,
        num_heads,
        num_layers,
        dropout
    ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            3,
            embed_dim
        )

        self.position_encoding = FourierPositionalEncoding2D(
            d_model=embed_dim,
            N=N,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_head = nn.Linear(embed_dim, 2)

    def causal_mask(self, size, device):

        mask = torch.triu(
            torch.ones(size, size, device=device),
            diagonal=1
        )

        mask = mask.masked_fill(
            mask == 1,
            float("-inf")
        )

        return mask

    def forward(self, x):

        h = self.token_embedding(x)

        h = self.position_encoding(h)

        mask = self.causal_mask(
            x.size(1),
            x.device
        )

        h = self.transformer(
            h,
            mask=mask
        )

        logits = self.output_head(h)

        return logits


# Main training loop with evaluation on held-out test set after each epoch

def train_model(model, train_loader, test_loader):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(EPOCHS):

        # Train

        model.train()

        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]"
        )

        for inp, target in train_bar:

            inp = inp.to(DEVICE)
            target = target.to(DEVICE)

            optimizer.zero_grad()

            logits = model(inp)

            loss = F.cross_entropy(
                logits.reshape(-1, 2),
                target.reshape(-1)
            )

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix(
                loss=loss.item()
            )

        avg_train_loss = train_loss / len(train_loader)

        # Evaluation

        model.eval()

        test_loss = 0.0

        print("\nEvaluating on held-out TEST set...")

        with torch.no_grad():

            test_bar = tqdm(
                test_loader,
                desc=f"Epoch {epoch+1}/{EPOCHS} [EVAL]"
            )

            for inp, target in test_bar:

                inp = inp.to(DEVICE)
                target = target.to(DEVICE)

                logits = model(inp)

                loss = F.cross_entropy(
                    logits.reshape(-1, 2),
                    target.reshape(-1)
                )

                test_loss += loss.item()

                test_bar.set_postfix(
                    loss=loss.item()
                )

        avg_test_loss = test_loss / len(test_loader)

        print("\n" + "-" * 60)
        print(f"Epoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.6f}")
        print(f"Test  Loss: {avg_test_loss:.6f}")
        print("-" * 60)

@torch.no_grad()
def sample_transformer(model, num_samples):

    print("\n" + "=" * 60)
    print("GENERATING TRANSFORMER SAMPLES")
    print("=" * 60)

    model.eval()

    samples = torch.zeros(
        (num_samples, SEQ_LEN),
        dtype=torch.long,
        device=DEVICE
    )

    generation_bar = tqdm(
        range(SEQ_LEN),
        desc="Autoregressive Sampling"
    )

    for t in generation_bar:

        prefix = samples[:, :t+1]

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):

            logits = model(prefix)

            probs = F.softmax(
                logits[:, -1],
                dim=-1
            )

        next_spin = torch.multinomial(
            probs,
            num_samples=1
        ).squeeze(-1)

        samples[:, t] = next_spin

    # map back:
    #
    # 0 -> -1
    # 1 -> +1
    #
    samples = 2 * samples - 1

    samples = samples.reshape(
        num_samples,
        N,
        N
    )

    return samples.cpu().numpy()

# Physics observables used for evaluation

def energy(config):

    E = 0

    for i in range(N):
        for j in range(N):

            s = config[i, j]

            neighbors = (
                config[(i + 1) % N, j]
                + config[i, (j + 1) % N]
            )

            E += -s * neighbors

    return E / (N * N)


def magnetization(config):

    return np.mean(config)


def correlation_function(configs, r_max=10):

    corrs = []

    for r in range(1, r_max + 1):

        vals = []

        for cfg in configs:

            shifted = np.roll(
                cfg,
                shift=r,
                axis=1
            )

            vals.append(
                np.mean(cfg * shifted)
            )

        corrs.append(np.mean(vals))

    return np.array(corrs)


# Evaluation

def evaluate(real_samples, generated_samples):

    print("\n" + "=" * 60)
    print("FINAL PHYSICS EVALUATION")
    print("=" * 60)

    real_E = np.array([
        energy(x)
        for x in tqdm(real_samples,
                      desc="Computing Real Energies")
    ])

    gen_E = np.array([
        energy(x)
        for x in tqdm(generated_samples,
                      desc="Computing Generated Energies")
    ])

    real_M = np.array([
        magnetization(x)
        for x in real_samples
    ])

    gen_M = np.array([
        magnetization(x)
        for x in generated_samples
    ])

    real_corr = correlation_function(real_samples)

    gen_corr = correlation_function(generated_samples)

    corr_error = np.mean(
        np.abs(
            np.log(np.abs(real_corr) + 1e-8)
            - np.log(np.abs(gen_corr) + 1e-8)
        )
    )

    print("\nEnergy Statistics")
    print("-" * 40)

    print("Real mean energy:",
          real_E.mean())

    print("Generated mean energy:",
          gen_E.mean())

    print("Absolute energy error:",
          abs(real_E.mean() - gen_E.mean()))

    print("\nMagnetization Statistics")
    print("-" * 40)

    print("Real variance:",
          real_M.var())

    print("Generated variance:",
          gen_M.var())

    print("Absolute magnetization variance error:",
          abs(real_M.var() - gen_M.var()))

    print("\nCorrelation Function")
    print("-" * 40)

    print("Real correlations:")
    print(real_corr)

    print("\nGenerated correlations:")
    print(gen_corr)

    print("\nMean log-correlation error:")
    print(corr_error)

if __name__ == "__main__":


    samples = generate_ising_dataset(
        NUM_TOTAL_SAMPLES
    )

    full_dataset = IsingDataset(samples)

    train_size = int(
        TRAIN_FRAC * len(full_dataset)
    )

    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size]
    )

    print("\n" + "=" * 60)
    print("DATASET SPLIT")
    print("=" * 60)

    print("Train samples:", len(train_dataset))
    print("Test  samples:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = IsingTransformer(
        N=N,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)


    train_model(
        model,
        train_loader,
        test_loader
    )


    generated_samples = sample_transformer(
        model,
        num_samples=1000
    )

    print("\nPreparing held-out test samples...")

    heldout_test_samples = []

    for inp, target in test_loader:

        spins = 2 * target.numpy() - 1

        spins = spins.reshape(
            spins.shape[0],
            N,
            N
        )

        heldout_test_samples.append(spins)

    heldout_test_samples = np.concatenate(
        heldout_test_samples,
        axis=0
    )

    evaluate(
        real_samples=heldout_test_samples[:1000],
        generated_samples=generated_samples
    )