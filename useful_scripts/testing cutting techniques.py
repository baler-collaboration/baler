import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

# Load data from NPZ file
with np.load('/home/kka011098/baler/workspaces/Anomaly_Detection/data/small_data2.npz') as npz_file:
    data = npz_file['data']

# Reduce the size of the dataset for testing purposes
max_data_size = 50000  # Adjust as needed
if len(data) > max_data_size:
    data = data[:max_data_size]

# Define chunk size based on dataset size
chunk_size = min(len(data), 1000)

# Function to process data in chunks with tqdm progress bar
def process_data_in_chunks(data, func, chunk_size):
    results = []
    total_chunks = len(data) // chunk_size + (1 if len(data) % chunk_size != 0 else 0)
    with tqdm(total=total_chunks, desc='Processing Data') as pbar:
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            results.append(func(chunk))
            pbar.update(1)
    return np.concatenate(results, axis=0)

# Standardize the data (compute mean and std over chunks)
scaler = StandardScaler()
def standardize_data(chunk):
    scaler.partial_fit(chunk)
    return scaler.transform(chunk)
X_scaled = process_data_in_chunks(data, standardize_data, chunk_size)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float)

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Train the Autoencoder
input_dim = X_tensor.shape[1]
latent_dim = 2  # Latent dimension for visualization
autoencoder = Autoencoder(input_dim, latent_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
num_epochs = 100
losses = []  # List to store losses for plotting
for epoch in tqdm(range(num_epochs), desc='Training Autoencoder'):
    for i in range(0, len(X_tensor), chunk_size):
        X_chunk = X_tensor[i:i+chunk_size]
        outputs = autoencoder(X_chunk)
        loss = criterion(outputs, X_chunk)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())  # Append current loss to list

# Extract latent representations (process in chunks)
def encode_data(chunk):
    return autoencoder.encoder(torch.tensor(chunk, dtype=torch.float)).detach().numpy()
encoded_X = process_data_in_chunks(X_scaled, encode_data, chunk_size)

# Perform dimensionality reduction for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Calculate min_samples based on a representative sample of the dataset
sample_size = min(1000, len(encoded_X))
neighbors = NearestNeighbors(n_neighbors=sample_size)
neighbors.fit(encoded_X)
distances, _ = neighbors.kneighbors(encoded_X)
avg_distance = np.mean(distances[:, 1:])
min_samples = max(5, int(sample_size * 0.01))  # Adjust the factor (0.01) based on your dataset

# Define the range of epsilon values for DBSCAN
neighbors = NearestNeighbors(n_neighbors=min_samples)
neighbors.fit(encoded_X)
distances, _ = neighbors.kneighbors(encoded_X)
reachability_distances = np.mean(distances[:, 1:], axis=1)  # Exclude the first nearest neighbor (distance = 0)

# Sort the reachability distances
sorted_distances = np.sort(reachability_distances)

# Identify the knee point in the reachability plot
knee_index = np.argmax(np.diff(sorted_distances, 2)) + 1
epsilon = sorted_distances[knee_index]

print("Selected Epsilon (Reachability Threshold):", epsilon)

# Clustering using DBSCAN on the encoded data
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(encoded_X)

# Separate data based on DBSCAN labels
anomalous_data = data[dbscan_labels == -1]
non_anomalous_data = data[dbscan_labels != -1]

# Ensure that sizes are in check (limiting the size to the length of the smallest class)
min_size = min(len(anomalous_data), len(non_anomalous_data))
anomalous_data = anomalous_data[:min_size]
non_anomalous_data = non_anomalous_data[:min_size]

# Plot loss curve and save to PDF
with PdfPages('plots.pdf') as pdf:
    plt.plot(losses)
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    pdf.savefig()
    plt.close()

    # Plot DBSCAN clustering and save to PDF
    plt.figure()
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis')
    plt.title('DBSCAN Clustering')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    pdf.savefig()
    plt.close()

    # Plot latent space and save to PDF
    plt.figure()
    plt.scatter(encoded_X[:, 0], encoded_X[:, 1], c=dbscan_labels, cmap='viridis')
    plt.title('Latent Space Representation')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    pdf.savefig()
    plt.close()

    # Plot actual data and save anomalous and non-anomalous data to NPZ files
    plt.figure()
    plt.hist(data.flatten(), bins=50, alpha=0.5, label='Original Data')
    plt.hist(non_anomalous_data.flatten(), bins=50, alpha=0.5, label='Non-anomalous Data')
    plt.legend()
    plt.title('Comparison of Original and Non-Anomalous Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    pdf.savefig()
    plt.close()
