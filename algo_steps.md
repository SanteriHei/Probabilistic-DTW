# Low-resolution alignment
- Input: A Feature sequence, containing the entire corpus with N frames.
1.1 Window the feature sequence to fixed-lenght sequence of L frames, Shift
S frames.
1.2 Downsample each feature-segment to M frames and concatenate -> (Md x 1) dim
   feature vector for each segment. (NOTE: Multiple feature vectors)
1.3 Calculate cosine distances between number of randomly sampled feature
vectors
1.4 Fit normal-cumulative distribution to the cosine distances
1.5 Calculate cosine distances between each vector pair, exclude the temporal
neighbourhood. Keep only pdf(dist) < alpha in memory.

# High-resolution alignment
2.1 Expand the candidate segments by +-E frames -> Allows alignment up to L+2E
frames
2.2 Calculate a regular affinity matrix of the original segment features using
the cosine-distances.
2.3 Transfer the pairwise distances (i.e. the affinity matrix) to
probabilities. (Use CDF of normal distribution). Estimate mean and variance
from random sample of paired feature vectors.
2.4 Use DTW to find minimum cost path through the resulting probability matrix
2.5 The alignment probability can be now measured as the product of the path
2.6 Calculate the likelihood ratio (LR) for all sub-paths
2.7 Discard a pair if its resulting alignment is shorter than L_min steps.

