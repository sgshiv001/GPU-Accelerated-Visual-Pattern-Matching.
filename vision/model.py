"""Digits dataset and the two thesis-inspired classifiers."""
import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from torch import nn

SEED = 42


def digits_split():
    data = load_digits()
    x = (data.images / 16).astype(np.float32)[:, None]
    y = data.target.astype(np.int64)
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
    return x, y, train, test


def hog(x):
    """Central differences, unsigned soft binning, four 4x4 cells, L2-Hys."""
    images = x[:, 0]
    gx, gy = np.zeros_like(images), np.zeros_like(images)
    gx[:, :, 1:-1] = images[:, :, 2:] - images[:, :, :-2]
    gy[:, 1:-1, :] = images[:, 2:, :] - images[:, :-2, :]
    magnitude = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) % np.pi) * (9 / np.pi) - 0.5
    lower = np.floor(angle).astype(int)
    frac = angle - lower
    cells = (np.arange(8)[:, None] // 4 * 2 + np.arange(8)[None, :] // 4) * 9
    features = np.zeros((len(x), 36), dtype=np.float64)
    row = np.arange(len(x))[:, None, None]
    np.add.at(features, (row, cells + lower % 9), magnitude * (1 - frac))
    np.add.at(features, (row, cells + (lower + 1) % 9), magnitude * frac)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    features = np.minimum(features, 0.2)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    return features


def svm_pipeline():
    return make_pipeline(PCA(n_components=0.95, svd_solver="full"), SVC(C=10, gamma="scale"))


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(self.conv1(x).relu(), 2)
        x = torch.nn.functional.max_pool2d(self.conv2(x).relu(), 2)
        return self.fc2(self.fc1(x.flatten(1)).relu())
