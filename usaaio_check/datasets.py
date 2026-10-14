import numpy as np, pandas as pd
def random_embeddings(N=300, d=100, seed=2026):
    """Clustered unit-ish vectors + token names so sim_dict has sensible answers."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((10, d))
    labels = rng.integers(0, 10, N)
    V = centers[labels] + 0.3 * rng.standard_normal((N, d))
    tokens = [f"tok{i:03d}" for i in range(N)]
    return tokens, V
def blobs(N=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, N)
    X = rng.standard_normal((N, 2)) + np.where(y[:, None] == 1, 2.0, -2.0)
    return X, y.astype(int)
def titanic_like(N=600, seed=2025):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Survived": rng.integers(0, 2, N), "Pclass": rng.integers(1, 4, N),
        "Sex": rng.choice(["male", "female"], N), "Age": rng.normal(30, 12, N).round(1),
        "SibSp": rng.integers(0, 4, N), "Parch": rng.integers(0, 3, N),
        "Fare": rng.gamma(2, 15, N).round(2), "Embarked": rng.choice(["S", "C", "Q"], N)})
    df.loc[rng.choice(N, min(40, max(1, N//15)), replace=False), "Age"] = np.nan   # missing values to drop
    return df
def iaio_kmeans_table():
    X = np.array([[1.90,0.97],[1.76,0.84],[2.32,1.63],[2.31,2.09],[1.14,2.11],
                  [5.02,3.02],[5.74,3.84],[2.25,3.47],[4.71,3.60],[3.17,4.96]])
    C0 = np.array([[1.90,0.97],[3.17,4.96]])
    return X, C0
