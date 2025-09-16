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
data_dir = os.path.join(parent_dir, 'data')

# If running in the root directory, use data directly
if os.path.exists('data'):
    data_dir = 'data'

# Set page configuration
st.set_page_config(
    page_title="LayerZero Sybil Detection System",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# css styling
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
    """Load all analysis data"""
    try:

        scores_file = os.path.join(data_dir, "final_sybil_scores.csv")
        tx_file = os.path.join(data_dir, "layerzero_transactions.csv")
        cluster_file = os.path.join(data_dir, "cluster_analysis_report.json")
        
        # Check if the file exists
        if not os.path.exists(scores_file):
            st.error(f"Risk scores file not found: {scores_file}")
            return None, None, None, False
            
        if not os.path.exists(tx_file):
            st.error(f"Transaction data file not found: {tx_file}")
            return None, None, None, False
            
        if not os.path.exists(cluster_file):
            st.error(f"Cluster analysis file not found: {cluster_file}")
            return None, None, None, False
        
        # Load data
        scores_df = pd.read_csv(scores_file)
        transactions_df = pd.read_csv(tx_file)
        
        # Fix time format conversion issue
        transactions_df['block_timestamp'] = pd.to_datetime(transactions_df['block_timestamp'], errors='coerce')
        
        # Check and remove invalid time data
        valid_timestamps = transactions_df['block_timestamp'].notna()
        if not valid_timestamps.all():
            print(f"Warning: Found {(~valid_timestamps).sum()} invalid time data")
            transactions_df = transactions_df[valid_timestamps]
        
        with open(cluster_file, "r", encoding='utf-8') as f:
            cluster_data = json.load(f)
            
        return scores_df, transactions_df, cluster_data, True
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info(f"📍 Current directory: {current_dir}")
        st.info(f"📍 副目錄: {parent_dir}")
        st.info(f"📍 數據目錄: {data_dir}")
        return None, None, None, False

def create_risk_score_charts(scores_df):
    """Create a risk score chart"""
    
    # 總分分佈圖
    fig_dist = px.histogram(
        scores_df, 
        x='composite_score', 
        nbins=10,
        title="🎯 女巫風險分數分佈",
        labels={'composite_score': '風險分數', 'count': '地址數量'},
        color_discrete_sequence=['#FF4B4B']
    )
    fig_dist.add_vline(
        x=scores_df['composite_score'].mean(), 
        line_dash="dash", 
        line_color="orange",
        annotation_text=f"平均分數: {scores_df['composite_score'].mean():.1f}"
    )
    fig_dist.update_layout(height=400)
    
    # 組件分數比較
    component_data = {
        '路徑相似性': scores_df['pathway_score'].mean(),
        '時間協調性': scores_df['temporal_score'].mean(),
        '行為相似性': scores_df['behavioral_score'].mean(),
        '網絡密度': scores_df['network_score'].mean()
    }
    
    fig_components = px.bar(
        x=list(component_data.keys()),
        y=list(component_data.values()),
        title="📊 各組件平均分數",
        labels={'x': '評分組件', 'y': '平均分數'},
        color=list(component_data.values()),
        color_continuous_scale='Reds'
    )
    fig_components.update_layout(height=400, showlegend=False)
    
    # 風險等級分佈
    risk_levels = scores_df['risk_level'].str.split(' - ').str[0].value_counts()
    colors = {'CRITICAL': '#FF0000', 'HIGH': '#FF6600', 'MEDIUM': '#FFCC00', 'LOW': '#66CC00', 'MINIMAL': '#00CC66'}
    pie_colors = [colors.get(level, '#CCCCCC') for level in risk_levels.index]
    
    fig_risk = px.pie(
        values=risk_levels.values,
        names=risk_levels.index,
        title="🚨 風險等級分佈",
        color_discrete_sequence=pie_colors
    )
    fig_risk.update_layout(height=400)
    
    return fig_dist, fig_components, fig_risk

def create_temporal_analysis(transactions_df):
    """創建時間分析圖表"""
    
    # 確保時間資料有效
    valid_df = transactions_df.dropna(subset=['block_timestamp']).copy()
    
    if len(valid_df) == 0:
        st.error("沒有有效的時間資料")
        return None, None
    
    # 改為月度活動熱力圖 - 更清晰的商業視角
    valid_df['year_month'] = valid_df['block_timestamp'].dt.to_period('M')
    monthly_activity = valid_df.groupby(['year_month', 'address']).size().reset_index(name='tx_count')
    monthly_activity['year_month_str'] = monthly_activity['year_month'].astype(str)
    
    # 按月份統計活躍地址數
    monthly_summary = valid_df.groupby('year_month').agg({
        'address': 'nunique',
        'guid': 'count'
    }).reset_index()
    monthly_summary.columns = ['month', 'active_addresses', 'total_transactions']
    monthly_summary['month_str'] = monthly_summary['month'].astype(str)
    
    # 活躍地址月度趨勢
    fig_timeline = px.bar(
        monthly_summary,
        x='month_str',
        y='active_addresses',
        title="📊 每月活躍地址數量 - 協調攻擊時間線",
        labels={'month_str': '月份', 'active_addresses': '活躍地址數量'},
        height=400,
        color='active_addresses',
        color_continuous_scale='Reds'
    )
    fig_timeline.update_layout(
        xaxis_title="月份",
        yaxis_title="活躍地址數量",
        xaxis={'tickangle': 45}
    )
    
    # 交易量月度分佈
    fig_frequency = px.line(
        monthly_summary,
        x='month_str',
        y='total_transactions',
        title="📈 每月交易量趨勢",
        labels={'month_str': '月份', 'total_transactions': '交易數量'},
        height=400,
        markers=True,
        line_shape='spline'
    )
    fig_frequency.update_traces(
        line=dict(color='#FF6B6B', width=3),
        marker=dict(size=8, color='#FF6B6B')
    )
    fig_frequency.update_layout(
        xaxis_title="月份",
        yaxis_title="交易數量",
        xaxis={'tickangle': 45}
    )
    
    return fig_timeline, fig_frequency

def create_pathway_analysis(transactions_df):
    """創建路徑分析圖表（顯示鏈名稱）"""

    # 鏈ID對照表
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

    # "115→130" "ZKSync → ZetaChain"
    def format_pathway(path):
        try:
            src, dst = map(int, path.split("→"))
            src_name = chain_id_map.get(src, f"EID {src}")
            dst_name = chain_id_map.get(dst, f"EID {dst}")
            return f"{src_name} → {dst_name}"
        except Exception:
            return path

    # ➤ 路徑頻率分析（Top 10）
    pathway_combos = transactions_df.apply(lambda row: f"{row['src_eid']}→{row['dst_eid']}", axis=1)
    pathway_counts = pathway_combos.value_counts().head(10).reset_index()
    pathway_counts.columns = ['pathway', 'count']
    pathway_counts['formatted'] = pathway_counts['pathway'].apply(format_pathway)

    fig_pathways = px.bar(
        pathway_counts,
        x='formatted',
        y='count',
        title="🛤️ TOP 10 跨鏈路徑使用頻率（鏈名稱顯示）",
        labels={'formatted': '跨鏈路徑（鏈名稱）', 'count': '使用次數'},
        height=400,
        color='count',
        color_continuous_scale='Blues'
    )
    fig_pathways.update_layout(showlegend=False, xaxis={'tickangle': 45})

    # ➤ 主要區塊鏈使用統計（來源/目的）— 也顯示鏈名稱
    chain_mapping = {
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
        30101: "Ethereum",
        30102: "BNB Chain",
        30106: "Avalanche",
        30109: "Polygon",
        30110: "Arbitrum",
        30111: "Optimism",
        30112: "Fantom",
        30116: "Core",
        30125: "Celo",
        30183: "Linea",
        30184: "Base",
        30195: "Mantle",
    }

    # Count the number of times each chain is used as a source chain and a target chain (Top 8)
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

    chain_stats = sorted(chain_stats, key=lambda x: x['total'], reverse=True)[:8]
    chain_df = pd.DataFrame(chain_stats)

    fig_chains = px.bar(
        chain_df,
        x='chain',
        y=['source_txs', 'destination_txs'],
        title="🔗 主要區塊鏈使用統計（鏈名稱）",
        labels={'value': '交易數量', 'chain': '區塊鏈'},
        height=400,
        color_discrete_map={'source_txs': '#FF6B6B', 'destination_txs': '#4ECDC4'}
    )
    fig_chains.update_layout(
        legend=dict(title="類型", orientation="h", y=1.02, x=0),
        xaxis={'tickangle': 45}
    )

    return fig_pathways, fig_chains



def create_network_graph(scores_df):
    """創建網絡關係圖 - 商業友好版本"""
    
    # 只顯示高風險地址的網絡關係
    high_risk_df = scores_df[scores_df['composite_score'] >= 75].copy()  # 只取高風險地址
    
    if len(high_risk_df) == 0:
        st.warning("沒有足夠的高風險地址來建立網絡圖")
        return None
    
    # 建立簡化的網絡圖
    G = nx.Graph()
    
    # 添加節點 - 按風險等級分組
    risk_groups = high_risk_df.groupby('risk_level')
    
    # 為不同風險等級設置不同顏色和大小
    risk_colors = {
        'CRITICAL - 極高女巫風險': '#FF0000',
        'HIGH - 高女巫風險': '#FF6600',
        'MEDIUM - 中等女巫風險': '#FFCC00'
    }
    
    # 計算布局
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
            
            # 按風險等級分層排列
            x = (i - group_size/2) * 0.3
            pos[addr_short] = (x, y_offset)
        
        y_offset -= 1.5
    
    # 添加邊 - 只連接風險分數相近的地址
    nodes = list(G.nodes())
    for i, node1 in enumerate(nodes):
        for node2 in nodes[i+1:]:
            score1 = G.nodes[node1]['score']
            score2 = G.nodes[node2]['score']
            if abs(score1 - score2) <= 5:  # 分數相近的地址連接
                G.add_edge(node1, node2)
    
    # 創建 Plotly 圖表
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # 準備節點資料
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
                    size=[15 + score/10 for score in scores],  # 分數越高，節點越大
                    color=color,
                    line=dict(width=2, color='black')
                ),
                text=nodes_in_level,
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                name=risk_level.split(' - ')[0],
                hovertemplate='地址: %{text}<br>風險分數: %{customdata}<extra></extra>',
                customdata=scores
            ))
    
    # 創建圖表
    fig = go.Figure()
    
    # 添加邊
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='lightgray'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # 添加節點
    for trace in node_trace:
        fig.add_trace(trace)
    
    fig.update_layout(
        title="高風險女巫地址關聯網絡",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text="節點大小表示風險分數，顏色表示風險等級",
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
    # 主標題
    st.markdown('<h1 class="main-header">🕵️ LayerZero Sybil Detection System</h1>', unsafe_allow_html=True)

    
    # 載入數據
    scores_df, transactions_df, cluster_data, data_loaded = load_data()
    
    if not data_loaded:
        st.error("❌ 無法載入數據")
        
        # 顯示一個基本的演示界面
        st.markdown("---")
        st.subheader("🎭 演示模式")
        st.info("以下是基於真實分析結果的演示數據")
        
        # 創建演示數據
        demo_metrics = {
            "檢測地址": "20",
            "時間範圍": "2022/12/30 - 2024/5/17",
            "資料來源": "LayerZero Scan API",
            "女巫確認率": "100%", 
            "平均風險分數": "86.2/100",
            "攻擊時間跨度": "1.5小時",
            "相同路徑地址": "20/20"
        }
        
        cols = st.columns(len(demo_metrics))
        for i, (key, value) in enumerate(demo_metrics.items()):
            with cols[i]:
                st.metric(key, value)
        
        return
    
    # 側邊欄
    
    # 分析概覽
    st.sidebar.markdown("### 分析概覽")
    st.sidebar.metric("總檢測地址", len(scores_df))
    st.sidebar.metric("總交易數", len(transactions_df))
    st.sidebar.metric("平均風險分數", f"{scores_df['composite_score'].mean():.1f}/100")
    st.sidebar.metric("最高風險分數", f"{scores_df['composite_score'].max()}/100")
    
    # 資料集資訊
    st.sidebar.markdown("### Dataset Information")
    st.sidebar.markdown("**Time**: 2022/12/30 - 2024/5/17")
    st.sidebar.markdown("**Sources**: LayerZero Scan API")
        
    # 風險等級統計
    risk_distribution = scores_df['risk_level'].str.split(' - ').str[0].value_counts()
    st.sidebar.markdown("### 🚨 風險等級分佈")
    for level, count in risk_distribution.items():
        percentage = (count / len(scores_df)) * 100
        if level == "HIGH":
            st.sidebar.markdown(f"🔴 **{level}**: {count} ({percentage:.1f}%)")
        elif level == "CRITICAL":
            st.sidebar.markdown(f"🚨 **{level}**: {count} ({percentage:.1f}%)")
        else:
            st.sidebar.markdown(f"🟡 **{level}**: {count} ({percentage:.1f}%)")
    
    # 主頁面標籤
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 風險分析", "⏰ 時間模式", "🛤️ 路徑分析", "📝 完整風險評估列表"])
    with tab1:
        
        # 創建風險分數圖表
        fig_dist, fig_components, fig_risk = create_risk_score_charts(scores_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_dist, use_container_width=True)
        with col2:
            st.plotly_chart(fig_risk, use_container_width=True)
        
        st.plotly_chart(fig_components, use_container_width=True)
        
        # TOP 風險地址表格
        st.subheader("🏆 TOP 10 最高風險地址")
        top_addresses = scores_df.nlargest(10, 'composite_score')[
            ['address', 'composite_score', 'risk_level', 'confidence', 
             'pathway_score', 'temporal_score', 'behavioral_score', 'network_score']
        ].round(1)
        
        st.dataframe(top_addresses, use_container_width=True)
    
    with tab2:
        
        # 創建時間分析圖表
        fig_timeline, fig_frequency = create_temporal_analysis(transactions_df)
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        st.plotly_chart(fig_frequency, use_container_width=True)
    
    with tab3:
        
        # 創建路徑分析圖表
        fig_pathways, fig_chains = create_pathway_analysis(transactions_df)
        
        st.plotly_chart(fig_pathways, use_container_width=True)
        st.plotly_chart(fig_chains, use_container_width=True)
    
    with tab4:

        # 添加下載按鈕
        csv_data = scores_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 下載完整報告 (CSV)",
            data=csv_data,
            file_name=f"sybil_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # 顯示完整表格
        st.dataframe(scores_df, use_container_width=True)


if __name__ == "__main__":
    main()