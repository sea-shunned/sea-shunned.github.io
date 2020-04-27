# Initial imports
from itertools import permutations, product
from collections import defaultdict
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform, cdist
import matplotlib.pyplot as plt
import seaborn as sns

def full_distances(data, metric="sqeuclidean"):
    return squareform(pdist(data, metric=metric))

def full_intraclusts(dists, N, cluster_size):
    # Calculate the intracluster variance
    a_vals = np.zeros((N,))
    # Loop over each cluster (exploit equal size)
    for start in range(0, N, cluster_size):
        # Get the final index for that cluster
        stop = start+cluster_size
        # Get the relevant part of the dist array
        clust_array = dists[start:stop, start:stop]
        # Divide by the cardinality
        a_vals[start:stop] = np.true_divide(
            np.sum(clust_array, axis=0),
            stop-start-1 # As we don't include d(i,i)
        )
    return a_vals

def full_interclusts(dists, N, K, cluster_size):
    # Calculate the intercluster variance
    # Get the permutations of cluster numbers
    perms = permutations(range(K), 2)
    # Initialize the b_values
    # Default to np.inf so we get a warning if it doesn't work
    b_vals = np.full((N, K), np.inf)
    # Calc each intercluster dist
    for c1, c2 in perms:
        # Get indices for distance array
        c1_start = c1*cluster_size
        c1_stop = c1_start+cluster_size
        c2_start = c2*cluster_size
        c2_stop = c2_start+cluster_size
        # Get the relevant part of the dist array
        clust_array = dists[c1_start:c1_stop, c2_start:c2_stop]
        # Select the minimum average distance
        b_vals[c1_start:c1_stop, c2] = np.mean(clust_array, axis=1)
    return b_vals

def full_silh(data, N, cluster_size, K):
    # Get the distances
    dists = full_distances(data)
    # Calculate the main terms
    a_vals = full_intraclusts(dists, N, cluster_size)
    b_vals = full_interclusts(dists, N, K, cluster_size)
    # Select the minimum average distance
    b_vec = b_vals.min(1)
    # Calculate the numerator and denominator
    top_term = b_vec - a_vals
    bottom_term = np.maximum(b_vec, a_vals)
    res = top_term / bottom_term
    # Handle singleton clusters
    res = np.nan_to_num(res)
    return np.mean(res), dists, a_vals, b_vals

def partial_distances(data, dists, cluster_size, changed_clusters, metric="sqeuclidean"):
    for i, changed in enumerate(changed_clusters):
        # Skip if unchanged
        if not changed:
            continue
        # Get the indices for the cluster
        start = i*cluster_size
        stop = start+cluster_size
        # Calculate the new block of pairwise matrices
        new_dists = cdist(
            data,
            data[start:stop, :],
            metric=metric
        )
        # Insert the new distances (in both as symmetric)
        dists[:, start:stop] = new_dists
        dists[start:stop, :] = new_dists.T
    return dists

def partial_intraclusts(a_vals, dists, N, cluster_size, changed_list):
    # Loop over each cluster (exploit equal size)
    for clust_num, changed in enumerate(changed_list):
        if not changed:
            continue
        # Get the indices for the cluster
        start = cluster_size * clust_num
        stop = start + cluster_size
        # Get the relevant part of the dist array
        clust_array = dists[start:stop, start:stop]
        # Divide by the cardinality
        a_vals[start:stop] = np.true_divide(
            np.sum(clust_array, axis=0),
            stop-start-1 # As we don't include d(i,i)
        )
    return a_vals

def partial_interclusts(b_vals, dists, K, cluster_size, changed_list):
    # Loop over the clusters that have changed
    for c1, changed in enumerate(changed_list):
        if changed:
            # Determine the relevant indices
            c1_start = c1*cluster_size
            c1_stop = c1_start+cluster_size
            # Loop over every other cluster
            for c2 in range(K):
                if c1 == c2:
                    continue
                # Get indices for distance array
                c2_start = c2*cluster_size
                c2_stop = c2_start+cluster_size
                # Get the relevant part of the dist array
                clust_array = dists[c1_start:c1_stop, c2_start:c2_stop]
                # Set the minimum average distance for the c1,c2 combo
                b_vals[c1_start:c1_stop, c2] = np.mean(clust_array, axis=1)
                # Set the minimum average distance for the c2,c1 combo
                b_vals[c2_start:c2_stop, c1] = np.mean(clust_array, axis=0)
    return b_vals

def partial_silh(data, N, cluster_size, K, changed_list=None, dists=None, a_vals=None, b_vals=None):
    if changed_list is None:
        changed_list = [True]*K
    # Get the distances
    if dists is None:
        dists = full_distances(data)
    else:
        dists = partial_distances(data, dists, cluster_size, changed_list)
    # Allows us to use this function for the first calc
    if a_vals is None:
        a_vals = np.zeros((N,))
    if b_vals is None:
        b_vals = np.full((N, K), np.inf)
    # Calculate the main terms
    a_vals = partial_intraclusts(a_vals, dists, N, cluster_size, changed_list)
    b_vals = partial_interclusts(b_vals, dists, K, cluster_size, changed_list)
    # Select the minimum average distance
    b_vec = b_vals.min(1)
    # Calculate the numerator and denominator
    top_term = b_vec - a_vals
    bottom_term = np.maximum(b_vec, a_vals)
    res = top_term / bottom_term
    # Handle singleton clusters
    res = np.nan_to_num(res)
    return np.mean(res), dists, a_vals, b_vals

def random_data(cluster_size, D, K):
    data_list = []
    for k in range(K):
        # Generate some random normal data
        data = np.random.multivariate_normal(
            mean=[np.random.rand() * 5 * k for _ in range(D)],
            cov=np.eye(D),
            size=cluster_size
        )
        # Append labels to the data
        # Note that this is only needed for the partial computation
        data = np.hstack((data, np.full((cluster_size, 1), k)))
        # Add the data to our list
        data_list.append(data)
    # Create a single array of the data
    full_data = np.concatenate(data_list)
    labels = full_data[:, -1]
    return full_data[:, :-1], labels

def modify_data(data, changed_list, cluster_size, D):
    new_data = data.copy()
    for i, changed in enumerate(changed_list):
        if changed:
            start = i*cluster_size
            stop = start + cluster_size
            new_data[start:stop] = np.random.multivariate_normal(
                mean=[np.random.rand() * np.random.randint(100, 1000) for _ in range(D)],
                cov=np.eye(D),
                size=cluster_size
            )
    return new_data

def test_full():
    # Set random seed
    np.random.seed(42)
    # Setup variables
    N, D, K = 1000, 2, 10
    cluster_size = int(N/K)
    data, labels = random_data(cluster_size, D, K)
    # Calculate the silhouette widths
    our_sw, _, _, _ = full_silh(data, N, cluster_size, K)
    sk_sw = silhouette_score(data, labels, metric='sqeuclidean')
    print(f"Our method:   {our_sw}")
    print(f"Scikit-learn: {sk_sw}")
    assert np.isclose(our_sw, sk_sw)

def test_partial():
    # Set random seed
    np.random.seed(42)
    # Setup variables
    N, D, K = 1000, 2, 10
    cluster_size = int(N/K)
    data, labels = random_data(cluster_size, D, K)
    changed_list = [True]*K
    # Need to first do the full calculation
    our_sw, dists, a_vals, b_vals = partial_silh(data, N, cluster_size, K, changed_list)
    # Compare with the full scikit calc to check
    sk_sw = silhouette_score(data, labels, metric='sqeuclidean')
    assert np.isclose(our_sw, sk_sw)
    # Drastically change the first cluster
    data[:100] = np.random.multivariate_normal(
        mean=[np.random.rand() * 1000 for _ in range(D)],
        cov=np.eye(D),
        size=cluster_size
    )
    # Drastically change the third cluster
    data[200:300] = np.random.multivariate_normal(
        mean=[np.random.rand() * 500 for _ in range(D)],
        cov=np.eye(D),
        size=cluster_size
    )
    # We changed the first and third cluster
    changed_list = [True, False, True, False, False]
    # Now partially recalculate the silhouette width
    our_sw, _, _, _ = partial_silh(data, N, cluster_size, K, changed_list, dists, a_vals, b_vals)
    sk_sw = silhouette_score(data, labels, metric='sqeuclidean')
    print(f"Partial method: {our_sw}")
    print(f"Full method:    {full_silh(data, N, cluster_size, K)[0]}")
    print(f"Scikit-learn:   {sk_sw}")
    assert np.isclose(our_sw, sk_sw)

def add_result(res, N, D, K, changes, method, time):
    res["# Examples"].append(N)
    res["Dimensionality"].append(D)
    res["Number of Clusters"].append(K)
    res["Proportion Changes"].append(changes)
    res["Method"].append(method)
    res["Time (s)"].append(time)
    return res

def time_comparison(Ns, Ds, Ks, changes_list):
    params = product(Ns, Ds, Ks, changes_list)
    res = defaultdict(list)
    repeats = 10
    # Loop over param combs
    for N, D, K, changes in params:
        print(N, D, K, sum(changes))
        cluster_size = int(N/K)
        data, labels = random_data(cluster_size, D, K)
        # Simulate expected mutations
        changed_list = [True if np.random.rand() < (changes/K) else False for _ in range(K)]
        # Just modify it the once so it doesn't interfere with timing
        new_data = modify_data(data, changed_list, cluster_size, D)

        for _ in range(repeats):
            # Full calculation
            start = time.time()
            # Run initial
            test = full_silh(data, N, cluster_size, K)
            # Run on mutated data
            full_sw, _, _, _ = full_silh(new_data, N, cluster_size, K)
            full_time = time.time() - start
            # Store the result
            res = add_result(res, N, D, K, changes, "Full", full_time)

            # Partial calculation
            start = time.time()
            # Run initial
            test2, dists, a_vals, b_vals = partial_silh(data, N, cluster_size, K)
            # Run on mutated data
            partial_sw, _, _, _ = partial_silh(new_data, N, cluster_size, K, changed_list, dists, a_vals, b_vals)
            partial_time = time.time() - start
            # Store the result
            res = add_result(res, N, D, K, changes, "Partial", partial_time)

            # sklearn calculation
            start = time.time()
            _ = silhouette_score(data, labels, metric='sqeuclidean')
            sklearn_sw = silhouette_score(new_data, labels, metric='sqeuclidean')
            sk_time = time.time() - start
            # Store the result
            res = add_result(res, N, D, K, changes, "sklearn", sk_time)
            
            # Check that all the results are the same
            assert np.isclose(full_sw, partial_sw) and np.isclose(partial_sw, sklearn_sw)

    df = pd.DataFrame(res)
    df.to_csv("sw_times.csv")
    print(df)
    return df

def plot_times(df):
    # Construct a facetgrid
    g = sns.FacetGrid(
        data=df.groupby(["# Examples", "Dimensionality", "Proportion Changes", "Method"])["Time (s)"].mean().reset_index(),
        row="# Examples",
        col="Dimensionality",
        hue="Method",
        margin_titles=True
    )
    # Plot the data
    g = g.map(plt.plot, "Proportion Changes", "Time (s)", marker=".").set(yscale="log")
    g = g.add_legend()
    # Save the results
    g.savefig("/home/cshand/Documents/my_website/static/img/sw_times.png", dpi=600)

if __name__ == "__main__":
    TEST = True

    if TEST:
        test_full()
        test_partial()

    # Main experiment
    # Set the magic number
    np.random.seed(42)
    # Set parameters to vary across
    Ns = [1000, 5000, 10000, 20000]
    Ds = [2, 50, 100]
    Ks = [50] # No need to vary if we change proportion
    changes_list = [1, 5, 10, 20, 30, 40, 50]
    # Load previously saved results
    if (Path.cwd() / "sw_times.csv").is_file():
        df = pd.read_csv("sw_times.csv")
    else:
        df = time_comparison(Ns, Ds, Ks, changes_list)
    # Plot the result
    plot_times(df)