# kmeans_archetypes_train_fixed.py
# Load CSV -> clean -> build per-author docs -> TFIDF -> SVD -> KMeans -> save joblib + assignments
# Requires: pip install pandas numpy scikit-learn joblib

import re
from typing import Optional, Dict

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


RANDOM_SEED = 42
MIN_CHARS_PER_TEXT = 1  # keep basically everything (change to 20 if you want)


def clean_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = x.replace("\n", " ").replace("\r", " ").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def load_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    # Standard comma-separated CSV
    # If your file is actually tab-separated, change sep="," to sep="\t".
    return pd.read_csv(path, sep=",", nrows=nrows, low_memory=False)


def build_user_docs(
    df: pd.DataFrame,
    max_users: Optional[int] = None,          # None = keep ALL users
    max_texts_per_user: Optional[int] = None  # None = keep ALL texts per user
) -> pd.DataFrame:
    """
    Returns DataFrame with columns: author, doc_text, n_items
    """
    needed = {"author", "body"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Need columns {needed}. Found: {list(df.columns)}")

    df = df.copy()

    # clean body
    df["body"] = df["body"].astype(str).map(clean_text)

    # drop junk rows
    df = df[df["body"].str.len() >= MIN_CHARS_PER_TEXT]
    df = df[~df["body"].isin(["[deleted]", "[removed]"])]
    df = df[df["author"].notna()]
    df = df[~df["author"].isin(["[deleted]", "AutoModerator"])]

    # count items per author (on the cleaned df)
    counts = df["author"].value_counts(dropna=True)

    # keep ALL users unless you explicitly cap max_users
    if max_users is not None:
        keep = set(counts.head(max_users).index.tolist())
        df = df[df["author"].isin(keep)]
        counts = df["author"].value_counts(dropna=True)

    # sort by time if present, so "first N" texts per user are consistent
    if "created_utc" in df.columns:
        df = df.sort_values("created_utc", ascending=True)

    # optionally cap texts per author
    if max_texts_per_user is not None:
        df["_rn"] = df.groupby("author").cumcount()
        df = df[df["_rn"] < max_texts_per_user]
        df = df.drop(columns=["_rn"])
        counts = df["author"].value_counts(dropna=True)

    # build one document per author
    grouped = df.groupby("author")["body"].apply(lambda s: " ".join(s.tolist()))
    user_docs = grouped.reset_index().rename(columns={"body": "doc_text"})
    user_docs["n_items"] = user_docs["author"].map(counts).fillna(0).astype(int)
    user_docs = user_docs.sort_values("n_items", ascending=False).reset_index(drop=True)

    return user_docs


def train_kmeans_archetypes(
    csv_path: str,
    k: int = 8,
    max_users: Optional[int] = None,          # None = ALL users
    max_texts_per_user: Optional[int] = None,   # set None to keep ALL rows per user
    model_out: str = "archetype_model.joblib",
    assignments_out: str = "user_clusters.csv",
) -> Dict:
    df = load_csv(csv_path)

    users = build_user_docs(
        df,
        max_users=max_users,
        max_texts_per_user=max_texts_per_user,
    )

    docs = users["doc_text"].tolist()
    if len(docs) < k:
        raise ValueError(f"Not enough users ({len(docs)}) to cluster into k={k} groups.")

    vectorizer = TfidfVectorizer(
        max_features=60_000,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english",
    )
    X_tfidf = vectorizer.fit_transform(docs)

    # n_components must be < n_features and <= n_samples-1 in practice
    n_components = min(200, max(2, X_tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
    X_svd = svd.fit_transform(X_tfidf)
    X_svd = normalize(X_svd)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED)
    clusters = kmeans.fit_predict(X_svd)

    out = users[["author", "n_items"]].copy()
    out["cluster_id"] = clusters
    out.to_csv(assignments_out, index=False)

    bundle = {
        "k": k,
        "vectorizer": vectorizer,
        "svd": svd,
        "kmeans": kmeans,
        "centroids": normalize(kmeans.cluster_centers_),
        "users_clustered": int(len(users)),
        "cluster_sizes": pd.Series(clusters).value_counts().sort_index().to_dict(),
        "max_users": max_users,
        "max_texts_per_user": max_texts_per_user,
    }
    joblib.dump(bundle, model_out)

    return bundle


if __name__ == "__main__":
    CSV_PATH = "canada_subreddit_comments.csv"  # change to your csv file
    K = 8

    bundle = train_kmeans_archetypes(
        csv_path=CSV_PATH,
        k=K,
        max_users=None,           # ✅ ALL users
        max_texts_per_user=None,    # set None to use ALL rows per user (can get huge)
        model_out="archetype_model.joblib",
        assignments_out="user_clusters.csv",
    )

    print("Saved model:", "archetype_model.joblib")
    print("Saved assignments:", "user_clusters.csv")
    print("Users clustered:", bundle["users_clustered"])
    print("Cluster sizes:", bundle["cluster_sizes"])
