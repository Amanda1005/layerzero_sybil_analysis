import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from datetime import datetime
import networkx as nx
import os
import sys

# Add subdirectories to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_dir = os.path.join(parent_dir, 'new_data')

# If running in the root directory, use data directly
if os.path.exists('new_data'):
    data_dir = 'new_data'

# Set page configuration
st.set_page_config(
    page_title="LayerZero Sybil Detection System", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #FF4B4B;
    }
    .alert-high {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #c62828;
        margin: 1rem 0;
    }
    .alert-success {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2e7d32;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all analysis data (from new_data/ or fallback to data/)"""
    try:
        base_dirs = ['new_data', 'data']
        for folder in base_dirs:
            scores_file = os.path.join(folder, "final_sybil_scores.csv")
            tx_file = os.path.join(folder, "layerzero_transactions.csv")
            cluster_file = os.path.join(folder, "cluster_analysis_report.json")

            if os.path.exists(scores_file) and os.path.exists(tx_file) and os.path.exists(cluster_file):
                scores_df = pd.read_csv(scores_file)
                transactions_df = pd.read_csv(tx_file)
                transactions_df['block_timestamp'] = pd.to_datetime(transactions_df['block_timestamp'], errors='coerce')
                transactions_df = transactions_df[transactions_df['block_timestamp'].notna()]  # Remove invalid timestamps

                with open(cluster_file, "r", encoding='utf-8') as f:
                    cluster_data = json.load(f)

                return scores_df, transactions_df, cluster_data, True, folder  # Return the folder used

        # None of the required data files found
        st.error("Required files not found (final_sybil_scores, transactions, cluster)")
        return None, None, None, False, None

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None, False, None



def create_risk_score_charts(scores_df):
    """Create risk score distribution and breakdown charts"""
    
    # Histogram of overall risk scores
    fig_dist = px.histogram(
        scores_df, 
        x='composite_score', 
        nbins=10,
        title="🎯 Risk Score Distribution",
        labels={'composite_score': 'Risk Score', 'count': 'Number of Addresses'},
        color_discrete_sequence=['#FF4B4B']
    )
    fig_dist.add_vline(
        x=scores_df['composite_score'].mean(), 
        line_dash="dash", 
        line_color="orange",
        annotation_text=f"Avg: {scores_df['composite_score'].mean():.1f}"
    )
    fig_dist.update_layout(height=400)
    
    # Component average scores
    component_data = {
        'Pathway Similarity': scores_df['pathway_score'].mean(),
        'Temporal Coordination': scores_df['temporal_score'].mean(),
        'Behavioral Similarity': scores_df['behavioral_score'].mean(),
        'Network Density': scores_df['network_score'].mean()
    }
    
    fig_components = px.bar(
        x=list(component_data.keys()),
        y=list(component_data.values()),
        title="Average Component Scores",
        labels={'x': 'Score Component', 'y': 'Average Score'},
        color=list(component_data.values()),
        color_continuous_scale='Reds'
    )
    fig_components.update_layout(height=400, showlegend=False)
    
    # Risk level distribution
    risk_levels = scores_df['risk_level'].str.split(' - ').str[0].value_counts()
    colors = {'CRITICAL': '#FF0000', 'HIGH': '#FF6600', 'MEDIUM': '#FFCC00', 'LOW': '#66CC00', 'MINIMAL': '#00CC66'}
    pie_colors = [colors.get(level, '#CCCCCC') for level in risk_levels.index]
    
    fig_risk = px.pie(
        values=risk_levels.values,
        names=risk_levels.index,
        title="Risk Level Distribution",
        color_discrete_sequence=pie_colors
    )
    fig_risk.update_layout(height=400)
    
    return fig_dist, fig_components, fig_risk


def create_temporal_analysis(transactions_df):
    """Create time-based analysis charts"""
    
    # Ensure timestamps are valid
    valid_df = transactions_df.dropna(subset=['block_timestamp']).copy()
    
    if len(valid_df) == 0:
        st.error("No valid timestamp data")
        return None, None
    
    # Convert to monthly view for clearer business-level insights
    valid_df['year_month'] = valid_df['block_timestamp'].dt.to_period('M')
    monthly_activity = valid_df.groupby(['year_month', 'address']).size().reset_index(name='tx_count')
    monthly_activity['year_month_str'] = monthly_activity['year_month'].astype(str)
    
    # Monthly summary of active addresses
    monthly_summary = valid_df.groupby('year_month').agg({
        'address': 'nunique',
        'guid': 'count'
    }).reset_index()
    monthly_summary.columns = ['month', 'active_addresses', 'total_transactions']
    monthly_summary['month_str'] = monthly_summary['month'].astype(str)
    
    # Bar chart of monthly active addresses
    fig_timeline = px.bar(
        monthly_summary,
        x='month_str',
        y='active_addresses',
        title="Monthly Active Addresses - Timeline of Coordinated Behavior",
        labels={'month_str': 'Month', 'active_addresses': 'Active Address Count'},
        height=400,
        color='active_addresses',
        color_continuous_scale='Reds'
    )
    fig_timeline.update_layout(
        xaxis_title="Month",
        yaxis_title="Active Address Count",
        xaxis={'tickangle': 45}
    )
    
    # Line chart of monthly transaction volume
    fig_frequency = px.line(
        monthly_summary,
        x='month_str',
        y='total_transactions',
        title="Monthly Transaction Volume Trend",
        labels={'month_str': 'Month', 'total_transactions': 'Transaction Count'},
        height=400,
        markers=True,
        line_shape='spline'
    )
    fig_frequency.update_traces(
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8, color='#FF6B6B')
    )
    fig_frequency.update_layout(
        xaxis_title="Month",
        yaxis_title="Transaction Count",
        xaxis={'tickangle': 45}
    )
    
    return fig_timeline, fig_frequency


def create_pathway_analysis(transactions_df):
    """Create pathway analysis charts (showing chain names)"""

    # Chain ID to Name Mapping
    chain_id_map = {
        102: "Binance",
        106: "Avalanche",
        109: "Polygon",
        110: "Arbitrum",
        111: "Optimism",
        112: "Fantom",
        115: "ZKSync",
        116: "Polygon zkEVM",
        125: "Metis",
        129: "ZKFair",
        130: "ZetaChain",
        150: "Manta",
        153: "Mode",
    }

    # Convert "115→130" to "ZKSync → ZetaChain"
    def format_pathway(path):
        try:
            src, dst = map(int, path.split("→"))
            src_name = chain_id_map.get(src, f"EID {src}")
            dst_name = chain_id_map.get(dst, f"EID {dst}")
            return f"{src_name} → {dst_name}"
        except Exception:
            return path

    # ➤ Top 10 Cross-chain Pathway Usage
    pathway_combos = transactions_df.apply(lambda row: f"{row['src_eid']}→{row['dst_eid']}", axis=1)
    pathway_counts = pathway_combos.value_counts().head(10).reset_index()
    pathway_counts.columns = ['pathway', 'count']
    pathway_counts['formatted'] = pathway_counts['pathway'].apply(format_pathway)

    fig_pathways = px.bar(
        pathway_counts,
        x='formatted',
        y='count',
        title="🛤️ Top 10 Cross-chain Pathway Usage (Chain Names)",
        labels={'formatted': 'Cross-chain Pathway', 'count': 'Usage Count'},
        height=400,
        color='count',
        color_continuous_scale='Blues'
    )
    fig_pathways.update_layout(showlegend=False, xaxis={'tickangle': 45})

    # ➤ Top Source/Destination Chain Usage
    chain_mapping = {
        102: "Binance", 106: "Avalanche", 109: "Polygon", 110: "Arbitrum", 111: "Optimism", 112: "Fantom",
        115: "ZKSync", 116: "Polygon zkEVM", 125: "Metis", 129: "ZKFair", 130: "ZetaChain", 150: "Manta", 153: "Mode",
        30101: "Ethereum", 30102: "BNB Chain", 30106: "Avalanche", 30109: "Polygon", 30110: "Arbitrum",
        30111: "Optimism", 30112: "Fantom", 30116: "Core", 30125: "Celo", 30183: "Linea", 30184: "Base", 30195: "Mantle"
    }

    # Count most frequently used source/destination chains (Top 8)
    src_chains = transactions_df['src_eid'].value_counts().head(8)
    dst_chains = transactions_df['dst_eid'].value_counts().head(8)

    chain_stats = []
    all_eids = set(src_chains.index) | set(dst_chains.index)
    for eid in all_eids:
        chain_name = chain_mapping.get(eid, f"EID {eid}")
        chain_stats.append({
            'chain': chain_name,
            'source_txs': int(src_chains.get(eid, 0)),
            'destination_txs': int(dst_chains.get(eid, 0)),
            'total': int(src_chains.get(eid, 0)) + int(dst_chains.get(eid, 0))
        })

    # Sort by total usage and keep Top 8
    chain_stats = sorted(chain_stats, key=lambda x: x['total'], reverse=True)[:8]
    chain_df = pd.DataFrame(chain_stats)

    fig_chains = px.bar(
        chain_df,
        x='chain',
        y=['source_txs', 'destination_txs'],
        title="Top Blockchain Usage (Source & Destination Chains)",
        labels={'value': 'Transaction Count', 'chain': 'Blockchain'},
        height=400,
        color_discrete_map={'source_txs': '#FF6B6B', 'destination_txs': '#4ECDC4'}
    )
    fig_chains.update_layout(
        legend=dict(title="Type", orientation="h", y=1.02, x=0),
        xaxis={'tickangle': 45}
    )

    return fig_pathways, fig_chains




def create_network_graph(scores_df):
    """Create network graph visualization (business-friendly version)"""
    
    # Filter: Only include high-risk addresses (score ≥ 75)
    high_risk_df = scores_df[scores_df['composite_score'] >= 75].copy()
    
    if len(high_risk_df) == 0:
        st.warning("Not enough high-risk addresses to generate the network graph.")
        return None

    # Initialize simplified graph
    G = nx.Graph()
    
    # Group nodes by risk level
    risk_groups = high_risk_df.groupby('risk_level')
    
    # Color settings for different risk levels
    risk_colors = {
        'CRITICAL - Extremely High Sybil Risk': '#FF0000',
        'HIGH - High Sybil Risk': '#FF6600',
        'MEDIUM - Medium Sybil Risk': '#FFCC00'
    }
    
    # Layout positions
    pos = {}
    y_offset = 0
    
    for risk_level, group in risk_groups:
        group_size = len(group)
        for i, (_, row) in enumerate(group.iterrows()):
            addr_short = row['address'][:8] + '...'
            G.add_node(addr_short,
                       score=row['composite_score'],
                       risk_level=risk_level,
                       full_address=row['address'])
            
            # Arrange nodes by risk level
            x = (i - group_size / 2) * 0.3
            pos[addr_short] = (x, y_offset)
        
        y_offset -= 1.5
    
    # Add edges: connect nodes with similar risk scores
    nodes = list(G.nodes())
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i + 1:]:
            score1 = G.nodes[node1]['score']
            score2 = G.nodes[node2]['score']
            if abs(score1 - score2) <= 5:
                G.add_edge(node1, node2)
    
    # Prepare edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Prepare node traces by risk level
    node_trace = []
    for risk_level, color in risk_colors.items():
        nodes_in_level = [n for n in G.nodes() if G.nodes[n]['risk_level'] == risk_level]
        if nodes_in_level:
            x_coords = [pos[node][0] for node in nodes_in_level]
            y_coords = [pos[node][1] for node in nodes_in_level]
            scores = [G.nodes[node]['score'] for node in nodes_in_level]
            
            node_trace.append(go.Scatter(
                x=x_coords, y=y_coords,
                mode='markers+text',
                marker=dict(
                    size=[15 + score / 10 for score in scores],
                    color=color,
                    line=dict(width=2, color='black')
                ),
                text=nodes_in_level,
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                name=risk_level.split(' - ')[0],
                hovertemplate='Address: %{text}<br>Risk Score: %{customdata}<extra></extra>',
                customdata=scores
            ))
    
    # Final figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='lightgray'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes
    for trace in node_trace:
        fig.add_trace(trace)
    
    fig.update_layout(
        title="High-Risk Sybil Address Network",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Node size = risk score, color = risk level",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def main():
    # Main Title
    st.markdown('<h1 class="main-header">🕵️ LayerZero Sybil Detection System</h1>', unsafe_allow_html=True)

    # Load data
    scores_df, transactions_df, cluster_data, data_loaded, data_folder = load_data()

    if not data_loaded:
        st.error("Failed to load data.")

        # Display basic demo mode
        st.markdown("---")
        st.subheader("Demo Mode")
        st.info("The following is a demo view based on actual analysis results.")

        # Create demo metrics
        demo_metrics = {
            "Detected Addresses": "20",
            "Time Range": "2022/12/30 - 2024/5/17",
            "Source": "LayerZero Scan API",
            "Sybil Match Rate": "100%", 
            "Average Risk Score": "86.2/100",
            "Attack Time Span": "1.5 hours",
            "Same Pathway Addresses": "20/20"
        }

        cols = st.columns(len(demo_metrics))
        for i, (key, value) in enumerate(demo_metrics.items()):
            with cols[i]:
                st.metric(label=key, value=value)
        return

    # Sidebar summary
    st.sidebar.markdown("### Analysis Summary")
    st.sidebar.metric("Total Addresses Analyzed", len(scores_df))
    st.sidebar.metric("Total Transactions", len(transactions_df))

    st.sidebar.markdown("### Dataset Information")
    st.sidebar.markdown("**Time Range**: 2022/12/30 - 2024/5/17")
    st.sidebar.markdown("**Source**: LayerZero Scan API")

    # Tabbed layout (6 tabs)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Risk Score Distribution", 
        "Component Scores", 
        "Monthly Active Addresses", 
        "Monthly Transaction Trend", 
        "Top Cross-Chain Pathways", 
        "Source/Destination Chains"
    ])

    with tab1:
        fig_dist, fig_components, fig_risk = create_risk_score_charts(scores_df)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_dist, use_container_width=True)
        with col2:
            st.plotly_chart(fig_risk, use_container_width=True)

    with tab2:
        _, fig_components, _ = create_risk_score_charts(scores_df)
        st.plotly_chart(fig_components, use_container_width=True)

    with tab3:
        fig_timeline, _ = create_temporal_analysis(transactions_df)
        st.plotly_chart(fig_timeline, use_container_width=True)

    with tab4:
        _, fig_frequency = create_temporal_analysis(transactions_df)
        st.plotly_chart(fig_frequency, use_container_width=True)

    with tab5:
        fig_pathways, _ = create_pathway_analysis(transactions_df)
        st.plotly_chart(fig_pathways, use_container_width=True)

    with tab6:
        _, fig_chains = create_pathway_analysis(transactions_df)
        st.plotly_chart(fig_chains, use_container_width=True)

if __name__ == "__main__":
    main()
