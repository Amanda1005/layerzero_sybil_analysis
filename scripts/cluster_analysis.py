import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime, timedelta
import json
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SybilClusterAnalyzer:
    def __init__(self):
        """女巫集群分析器"""
        self.transactions_df = None
        self.address_features = None
        self.similarity_graph = None
        self.clusters = {}
        
    def load_data(self, tx_file="data/layerzero_transactions.csv"):
        """載入交易資料"""
        try:
            self.transactions_df = pd.read_csv(tx_file)
            logging.info(f"📖 載入 {len(self.transactions_df)} 筆交易，{self.transactions_df['address'].nunique()} 個地址")
            
            # 資料預處理
            self.transactions_df['block_timestamp'] = pd.to_datetime(self.transactions_df['block_timestamp'])
            self.transactions_df['created'] = pd.to_datetime(self.transactions_df['created'])
            
            return True
        except FileNotFoundError:
            logging.error(f"❌ 找不到交易檔案: {tx_file}")
            return False
    
    def extract_behavioral_features(self):
        """提取地址行為特徵"""
        logging.info("🔍 提取地址行為特徵...")
        
        features_list = []
        
        for address in self.transactions_df['address'].unique():
            addr_txs = self.transactions_df[self.transactions_df['address'] == address]
            
            # 基本統計特徵
            basic_features = {
                'address': address,
                'tx_count': len(addr_txs),
                'unique_src_chains': addr_txs['src_eid'].nunique(),
                'unique_dst_chains': addr_txs['dst_eid'].nunique(),
                'unique_pathways': addr_txs['pathway_id'].nunique(),
            }
            
            # 時間特徵
            timestamps = addr_txs['block_timestamp'].dropna()
            if len(timestamps) > 1:
                time_diffs = timestamps.diff().dropna()
                time_features = {
                    'first_tx_time': timestamps.min(),
                    'last_tx_time': timestamps.max(),
                    'tx_timespan_hours': (timestamps.max() - timestamps.min()).total_seconds() / 3600,
                    'avg_interval_minutes': time_diffs.dt.total_seconds().mean() / 60,
                    'std_interval_minutes': time_diffs.dt.total_seconds().std() / 60,
                    'min_interval_seconds': time_diffs.dt.total_seconds().min(),
                    'max_interval_seconds': time_diffs.dt.total_seconds().max(),
                }
            else:
                time_features = {
                    'first_tx_time': timestamps.min() if len(timestamps) > 0 else None,
                    'last_tx_time': timestamps.max() if len(timestamps) > 0 else None,
                    'tx_timespan_hours': 0,
                    'avg_interval_minutes': 0,
                    'std_interval_minutes': 0,
                    'min_interval_seconds': 0,
                    'max_interval_seconds': 0,
                }
            
            # 路徑模式特徵
            pathway_patterns = {
                'most_common_src_eid': addr_txs['src_eid'].mode().iloc[0] if len(addr_txs['src_eid'].mode()) > 0 else None,
                'most_common_dst_eid': addr_txs['dst_eid'].mode().iloc[0] if len(addr_txs['dst_eid'].mode()) > 0 else None,
                'src_eid_entropy': self._calculate_entropy(addr_txs['src_eid']),
                'dst_eid_entropy': self._calculate_entropy(addr_txs['dst_eid']),
            }
            
            # nonce 模式
            nonces = addr_txs['nonce'].dropna()
            nonce_features = {
                'nonce_range': nonces.max() - nonces.min() if len(nonces) > 0 else 0,
                'nonce_gaps': len(nonces) - (nonces.max() - nonces.min() + 1) if len(nonces) > 0 else 0,
                'sequential_nonces': (nonces.diff().dropna() == 1).sum() if len(nonces) > 1 else 0,
            }
            
            # 合併所有特徵
            features = {**basic_features, **time_features, **pathway_patterns, **nonce_features}
            features_list.append(features)
        
        self.address_features = pd.DataFrame(features_list)
        logging.info(f"✅ 提取完成，共 {len(self.address_features)} 個地址的 {len(self.address_features.columns)-1} 個特徵")
        
        return self.address_features
    
    def _calculate_entropy(self, series):
        """計算資訊熵"""
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def calculate_address_similarity(self):
        """計算地址間的相似度"""
        logging.info("🔗 計算地址間相似度...")
        
        if self.address_features is None:
            logging.error("❌ 請先提取特徵")
            return None
        
        # 選擇數值特徵
        numeric_features = self.address_features.select_dtypes(include=[np.number]).drop(columns=['address'], errors='ignore')
        
        # 標準化特徵
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(numeric_features.fillna(0))
        
        # 計算餘弦相似度
        similarity_matrix = cosine_similarity(features_scaled)
        
        # 建立相似度圖
        self.similarity_graph = nx.Graph()
        addresses = self.address_features['address'].tolist()
        
        # 添加節點
        for i, addr in enumerate(addresses):
            self.similarity_graph.add_node(addr, features=features_scaled[i])
        
        # 添加邊（相似度閾值）
        similarity_threshold = 0.95  # 高相似度閾值
        for i in range(len(addresses)):
            for j in range(i+1, len(addresses)):
                similarity = similarity_matrix[i, j]
                if similarity >= similarity_threshold:
                    self.similarity_graph.add_edge(
                        addresses[i], 
                        addresses[j], 
                        weight=similarity,
                        similarity=similarity
                    )
        
        logging.info(f"✅ 相似度圖建立完成：{len(self.similarity_graph.nodes)} 個節點，{len(self.similarity_graph.edges)} 條邊")
        return similarity_matrix
    
    def detect_temporal_clusters(self, time_window_minutes=30):
        """檢測時間集群"""
        logging.info(f"⏰ 檢測時間集群（窗口：{time_window_minutes} 分鐘）...")
        
        temporal_clusters = []
        
        # 按時間排序所有交易
        all_txs = self.transactions_df.sort_values('block_timestamp')
        
        current_cluster = []
        cluster_start_time = None
        
        for _, tx in all_txs.iterrows():
            tx_time = tx['block_timestamp']
            
            if cluster_start_time is None:
                # 開始新集群
                cluster_start_time = tx_time
                current_cluster = [tx]
            elif (tx_time - cluster_start_time).total_seconds() <= time_window_minutes * 60:
                # 在時間窗口內
                current_cluster.append(tx)
            else:
                # 超出時間窗口，保存當前集群並開始新集群
                if len(current_cluster) >= 2:  # 至少2筆交易才算集群
                    temporal_clusters.append({
                        'cluster_id': len(temporal_clusters),
                        'start_time': cluster_start_time,
                        'end_time': current_cluster[-1]['block_timestamp'],
                        'addresses': list(set([tx['address'] for tx in current_cluster])),
                        'transaction_count': len(current_cluster),
                        'unique_addresses': len(set([tx['address'] for tx in current_cluster]))
                    })
                
                cluster_start_time = tx_time
                current_cluster = [tx]
        
        # 處理最後一個集群
        if len(current_cluster) >= 2:
            temporal_clusters.append({
                'cluster_id': len(temporal_clusters),
                'start_time': cluster_start_time,
                'end_time': current_cluster[-1]['block_timestamp'],
                'addresses': list(set([tx['address'] for tx in current_cluster])),
                'transaction_count': len(current_cluster),
                'unique_addresses': len(set([tx['address'] for tx in current_cluster]))
            })
        
        self.clusters['temporal'] = temporal_clusters
        logging.info(f"✅ 發現 {len(temporal_clusters)} 個時間集群")
        
        return temporal_clusters
    
    def detect_pathway_clusters(self):
        """檢測相同路徑集群"""
        logging.info("🛤️ 檢測路徑模式集群...")
        
        # 分析路徑序列模式
        pathway_clusters = defaultdict(list)
        
        for address in self.transactions_df['address'].unique():
            addr_txs = self.transactions_df[self.transactions_df['address'] == address].sort_values('block_timestamp')
            
            # 建立路徑序列
            pathway_sequence = []
            for _, tx in addr_txs.iterrows():
                pathway_sequence.append(f"{tx['src_eid']}->{tx['dst_eid']}")
            
            # 使用路徑序列作為集群key
            pathway_pattern = "|".join(pathway_sequence)
            pathway_clusters[pathway_pattern].append(address)
        
        # 轉換為列表格式
        pathway_cluster_list = []
        for pattern, addresses in pathway_clusters.items():
            if len(addresses) >= 2:  # 至少2個地址才算集群
                pathway_cluster_list.append({
                    'cluster_id': len(pathway_cluster_list),
                    'pattern': pattern,
                    'addresses': addresses,
                    'size': len(addresses)
                })
        
        self.clusters['pathway'] = pathway_cluster_list
        logging.info(f"✅ 發現 {len(pathway_cluster_list)} 個路徑模式集群")
        
        return pathway_cluster_list
    
    def detect_similarity_clusters(self, eps=0.1, min_samples=2):
        """使用 DBSCAN 檢測相似度集群"""
        logging.info("🎯 使用 DBSCAN 檢測行為相似集群...")
        
        if self.address_features is None:
            logging.error("❌ 請先提取特徵")
            return []
        
        # 準備數值特徵
        numeric_features = self.address_features.select_dtypes(include=[np.number])
        features_scaled = StandardScaler().fit_transform(numeric_features.fillna(0))
        
        # DBSCAN 聚類
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        # 整理聚類結果
        similarity_clusters = []
        for cluster_id in set(cluster_labels):
            if cluster_id != -1:  # 排除噪音點
                cluster_addresses = self.address_features[cluster_labels == cluster_id]['address'].tolist()
                similarity_clusters.append({
                    'cluster_id': cluster_id,
                    'addresses': cluster_addresses,
                    'size': len(cluster_addresses)
                })
        
        self.clusters['similarity'] = similarity_clusters
        logging.info(f"✅ 發現 {len(similarity_clusters)} 個行為相似集群")
        
        return similarity_clusters
    
    def generate_cluster_report(self):
        """生成集群分析報告"""
        logging.info("📋 生成集群分析報告...")
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_addresses': self.address_features['address'].nunique() if self.address_features is not None else 0,
            'total_transactions': len(self.transactions_df) if self.transactions_df is not None else 0,
            'clusters': {}
        }
        
        for cluster_type, clusters in self.clusters.items():
            report['clusters'][cluster_type] = {
                'count': len(clusters),
                'details': clusters
            }
        
        # 儲存報告
        os.makedirs("data", exist_ok=True)
        with open("data/cluster_analysis_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 也儲存為 CSV 格式的摘要
        summary_data = []
        for cluster_type, clusters in self.clusters.items():
            for cluster in clusters:
                summary_data.append({
                    'cluster_type': cluster_type,
                    'cluster_id': cluster['cluster_id'],
                    'size': cluster.get('size', len(cluster.get('addresses', []))),
                    'addresses': ','.join(cluster.get('addresses', [])),
                    'pattern': cluster.get('pattern', ''),
                    'risk_score': len(cluster.get('addresses', [])) * 10  # 初步風險分數
                })
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            summary_df.to_csv("data/cluster_summary.csv", index=False)
            logging.info("💾 集群摘要已儲存至 data/cluster_summary.csv")
        
        return report
    
    def print_analysis_summary(self):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("🕵️ 女巫集群分析報告")
        print("="*60)
        
        if self.address_features is not None:
            print(f"📊 總地址數: {self.address_features['address'].nunique()}")
            print(f"📈 總交易數: {len(self.transactions_df)}")
        
        for cluster_type, clusters in self.clusters.items():
            print(f"\n🔍 {cluster_type.upper()} 集群:")
            if clusters:
                for cluster in clusters[:5]:  # 只顯示前5個
                    size = cluster.get('size', len(cluster.get('addresses', [])))
                    print(f"  集群 {cluster['cluster_id']}: {size} 個地址")
                    if 'pattern' in cluster:
                        print(f"    模式: {cluster['pattern'][:100]}...")
                
                if len(clusters) > 5:
                    print(f"  ... 還有 {len(clusters)-5} 個集群")
            else:
                print("  未發現集群")
        
        print("="*60)

def main():
    """主執行函數"""
    
    # 初始化分析器
    analyzer = SybilClusterAnalyzer()
    
    # 載入資料
    if not analyzer.load_data():
        return
    
    # 提取特徵
    features = analyzer.extract_behavioral_features()
    
    # 計算相似度
    similarity_matrix = analyzer.calculate_address_similarity()
    
    # 檢測各種類型的集群
    temporal_clusters = analyzer.detect_temporal_clusters(time_window_minutes=30)
    pathway_clusters = analyzer.detect_pathway_clusters()
    similarity_clusters = analyzer.detect_similarity_clusters(eps=0.1, min_samples=2)
    
    # 生成報告
    report = analyzer.generate_cluster_report()
    
    # 打印摘要
    analyzer.print_analysis_summary()
    
    logging.info("🎯 集群分析完成！")

if __name__ == "__main__":
    main()