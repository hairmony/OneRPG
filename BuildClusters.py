import json
import re
import pandas as pd

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\n", " ").replace("\r", " ").strip()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

COMMENTS_PATH = "canada_subreddit_comments.csv"
CLUSTERS_PATH = "user_clusters.csv"
OUT_PATH = "cluster_profiles.json"

print("Loading data...")
comments = pd.read_csv(COMMENTS_PATH, low_memory=False, encoding="latin1")
clusters = pd.read_csv(CLUSTERS_PATH)

# Fix encoding
comments["body"] = (
    comments["body"]
    .astype(str)
    .str.encode("latin1", errors="ignore")
    .str.decode("utf-8", errors="ignore")
)

# Join cluster labels onto comments
df = comments.merge(
    clusters[["author", "cluster_id"]],
    on="author",
    how="inner"
)

# Clean + filter
df["body"] = df["body"].map(clean_text)
df = df[df["body"].str.len() >= 40]
df = df[~df["body"].isin(["[deleted]", "[removed]"])]

# Simple features
df["len"] = df["body"].str.len()
df["has_q"] = df["body"].str.contains(r"\?", regex=True)

profiles = {}

print("Building cluster profiles...")
for cid, g in df.groupby("cluster_id"):
    examples = g.sample(n=min(12, len(g)), random_state=42)["body"].tolist()
    examples = [x[:300] for x in examples]  # keep prompts small

    profiles[int(cid)] = {
        "cluster_id": int(cid),
        "n_comments": int(len(g)),
        "avg_length": float(g["len"].mean()),
        "question_ratio": float(g["has_q"].mean()),
        "example_comments": examples
    }

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(profiles, f, ensure_ascii=False, indent=2)

print("Saved:", OUT_PATH)
