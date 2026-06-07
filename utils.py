import pandas as pd
import torch
import networkx as nx

from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data

MODEL_NAME = "all-MiniLM-L6-v2"

def load_reviews(csv_path):
    return pd.read_csv(csv_path)

def generate_embeddings(df):
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["review_text"].tolist(),
        convert_to_numpy=True
    )
    return embeddings

def build_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        user = row["user_id"]
        product = row["product_id"]

        G.add_edge(user, product)
    return G

def create_edge_index(df):
    edges = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            same_product = (
                df.iloc[i]["product_id"]
                ==
                df.iloc[j]["product_id"]
            )
            if same_product:
                edges.append([i, j])
                edges.append([j, i])
    if len(edges) == 0:
        edges.append([0, 0])
    edge_index = torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()
    return edge_index

def create_pyg_data(df):
    embeddings = generate_embeddings(df)
    x = torch.tensor(
        embeddings,
        dtype=torch.float
    )
    edge_index = create_edge_index(df)
    y = torch.tensor(
        df["fake"].values,
        dtype=torch.long
    )
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )
    return data

def reviewer_risk(df):
    counts = (
        df.groupby("user_id")
        .size()
        .reset_index(name="reviews")
    )
    return counts                

