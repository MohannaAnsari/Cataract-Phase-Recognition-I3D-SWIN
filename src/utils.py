import os, random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

def seed_everything(seed=42):
    # Make results reproducible.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def plot_phase_distribution(df, save_path=None):
    # Plot number of segments per phase.
    plt.figure(figsize=(10,5))
    sns.countplot(y='Phase', data=df, order=df['Phase'].value_counts().index)
    plt.title("Number of Segments per Phase")
    plt.xlabel("Count"); plt.ylabel("Phase")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_transition_matrix(matrix, save_path=None):
    # Visualize phase transition matrix.
    plt.figure(figsize=(8,6))
    sns.heatmap(matrix, cmap='Blues')
    plt.title("Phase Transition Matrix")
    plt.xlabel("Next Phase"); plt.ylabel("Current Phase")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
