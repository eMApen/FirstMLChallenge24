# 2024 Wireless Algorithm Contest: Preliminary Round

This repository contains the implementation and results of the preliminary round for the 2024 Wireless Algorithm
Contest. The goal was to map channel state information (CSI) to spatial coordinates using machine learning techniques,
guided by the methodology described in the IEEE TCOM
article [DOI: 10.1109/TCOMM.2022.3205152](https://ieeexplore.ieee.org/abstract/document/10495336).

## Project Structure

- `0. LoadingCSI.ipynb`: Jupyter notebook for loading the CSI data.
- `1. CalculateMatrix.ipynb`: Jupyter notebook for calculating the dissimilarity matrices.
- `2. TripleTrain.ipynb`: Jupyter notebook for training the Triplet Network.
- `3. TripleTest.ipynb`: Jupyter notebook for testing the Triplet Network.
- `absdissmilarity.py`: Script for calculating absolute dissimilarity.
- `anchorplt.py`: Script for plotting anchor points.
- `dissimilarity.py`: Script for calculating the CS dissimilarity matrix.
- `dataset.py`: Script for handling dataset operations.
- `geodesic.py`: Script for calculating geodesic distances.
- `mapping.py`: Script for mapping embeddings to coordinates.
- `tritrain.py`: Script for training the Triplet Network.
- `CompetitionData1/`: Directory containing the competition dataset.
- `Example/`: Directory containing example data and scripts.
- `Readme.md`: This README file.

## Methodology

### 1. CS Dissimilarity Calculation

The CS dissimilarity matrix is calculated based on the provided channel data. The dissimilarity between two points is
defined as:

\[ d_{\mathrm{CS}, i, j}=\sum_{b=1}^B \sum_{n=1} ^{N_{sub}}\left(1-\frac{\left|\sum_{m=1}^M\left(\tilde{\mathbf{H}}_{b,
m, n}^{(i)}\right)^* \tilde{\mathbf{H}}_{b, m, n}^{(j)}\right|^2}{\left(\sum_{m=1}^M\left|\tilde{\mathbf{H}}_{b, m,
n}^{(i)}\right|^2\right)\left(\sum_{m=1}^M\left|\tilde{\mathbf{H}}_{b, m, n}^{(j)}\right|^2\right)}\right) \]

This formula is adapted from the method described in the referenced IEEE TCOM article.

### 2. k-NN Graph Construction

A k-nearest neighbors graph is constructed from the dissimilarity matrix. The graph is then used to compute the shortest
path distances between points using Dijkstra's algorithm.

### 3. Triplet Network Training

A Triplet Network is trained to map CSI to spatial coordinates. The network architecture consists of several fully
connected layers with ReLU activations and batch normalization. The Triplet Loss function ensures that the distance
between anchor and positive samples is minimized, while the distance between anchor and negative samples is maximized
with a margin.

## Training Results

- **Loss Function**: Triplet Loss with a margin of 1.0.
- **Optimizer**: Adam optimizer with a learning rate of 0.001.
- **Batch Size**: 64.
- **Epochs**: 100.

### Performance Metrics

- **Training Loss**: The training loss decreased to a certain value and then plateaued, indicating that the model has
  learned a meaningful representation of the data.
- **Validation Loss**: The validation loss followed a similar trend to the training loss, suggesting that the model is
  not overfitting.

## Conclusion

The trained model successfully maps CSI to spatial coordinates. However, in the preliminary round, the model achieved
44th place and did not advance to the final round. We faced challenges due to our unfamiliarity with neural network
tuning and implementation, which resulted in suboptimal training performance. Given the more complex channel environment
compared to the reference paper, a more sophisticated network architecture would likely be more effective. Future work
includes refining the model architecture, tuning hyperparameters, and exploring alternative loss functions to further
improve performance.

## Acknowledgements

This work is part of the 2024 Wireless Algorithm Contest. We thank the organizers for providing the dataset and
evaluation framework. The methodology was guided by the IEEE TCOM article DOI: 10.1109/TCOMM.2022.3205152