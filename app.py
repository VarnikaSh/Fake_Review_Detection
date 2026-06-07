import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import torch
import plotly.express as px
import plotly.graph_objects as go

from torch_geometric.nn import SAGEConv

from utils import(
    load_reviews,
    create_pyg_data
)

st.set_page_config(
    page_title="FakeReviewShield AI",
    page_icon="🛡️",
    layout="wide"
)
st.title("🛡️ FakeReviewShield AI")

class FraudGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(
            384,
            128
        )
        self.conv2 = SAGEConv(
            128,
            64
        )
        self.conv3 = SAGEConv(
            64, 
            2
        )

    def forward(
            self,
            x,
            edge_index
    ):
        x = self.conv1(
            x,
            edge_index
        )
        x = torch.relu(x)
        x = self.conv2(
            x,
            edge_index
        )
        x = torch.relu(x)
        x = self.conv3(
            x,
            edge_index
        )
        return x

uploaded_file = st.file_uploader(
    "Upload Reviews CSV",
    type=["csv"]
)
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    data = create_pyg_data(df)

    model = FraudGNN()
    model.load_state_dict(
        torch.load(
            "fraud_gnn.pt",
            map_location="cpu"
        )
    )
    model.eval()
    with torch.no_grad():
        out = model(
            data.x,
            data.edge_index
        )
        probabilities = (
            torch.softmax(
                out,
                dim=1
            )[:,1]
            .numpy()
        )
    df["fraud_probability"] = (
        probabilities * 100
    )

    st.subheader(
        "📊 Dashboard Metrics"
    )
    col1,col2,col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Reviews",
            len(df)
        )
    with col2:
        st.metric(
            "Users",
            df["user_id"].nunique()
        )
    with col3:
        st.metric(
            "Products",
            df["product_id"].nunique()
        )

    st.subheader(
        "⚠️ Fraud Predictions"
    )
    st.dataframe(
        df[
            [
                "user_id",
                "product_id",
                "fraud_probability"
            ]
        ]
        .sort_values(
            "fraud_probability",
            ascending=False
        )
    )

    st.subheader(
        "🚨 High Risk Reviews"
    )
    high_risk = df[
        df["fraud_probability"] > 70
    ]
    st.dataframe(high_risk)

    st.subheader(
        "👤 Reviewer Risk Analysis"
    )
    reviewer_stats = (
        df.groupby("user_id")
        .agg({
            "fraud_probability":"mean"
        })
        .reset_index()
    )

    fig = px.bar(
        reviewer_stats,
        x="user_id",
        y="fraud_probability",
        title="Average Fraud Score by Reviewer"
    )
    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader(
        "📦 Product Risk Analysis"
    )
    product_stats = (
        df.groupby("product_id")
        .agg({
            "fraud_probability":"mean"
        })
        .reset_index()
    )
    fig2 = px.bar(
        product_stats,
        x="product_id",
        y="fraud_probability",
        title="Average Fraud Score by Product"
    )
    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.subheader(
        "🕸️ Community Detection"
    )

    G = nx.Graph()

    for _, row in df.iterrows():
        G.add_edge(
            row["user_id"],
            row["product_id"]
        )                  
    communities = list(
        nx.connected_components(G)
    )
    st.metric(
        "Communities Found",
        len(communities)
    )  
    for i, community in enumerate(
        communities[:5]
    ):
        st.write(
            f"Community {i+1}:",
            list(community)
        )

    st.subheader(
        "🌐 Reviewer-Product Network"
    )
    pos = nx.spring_layout(
        G,
        seed=42
    )
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0,y0 = pos[edge[0]]
        x1,y1 = pos[edge[1]]
        edge_x.extend(
            [x0,x1,None]
        )       
        edge_y.extend(
            [y0,y1,None]
        )
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines"
    )
    node_x = []
    node_y = []
    for node in  G.nodes():
        x,y = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(G.nodes())
    )
    fig_graph = go.Figure(
        data=[
            edge_trace,
            node_trace
        ]
    )
    st.plotly_chart(
        fig_graph,
        use_container_width=True
    )

    st.subheader(
        "📥 Export Results"
    )     
    csv = df.to_csv(
        index=False
    ) 
    st.download_button(
        "Download Fraud Report",
        csv,
        file_name="fraud_report.csv"
    )


