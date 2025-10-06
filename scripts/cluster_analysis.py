# cluster_analysis.py

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime
import json
import os
from collections import defaultdict

# 日誌設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SybilClusterAnalyzer:
    def __init__(self):
        self.transactions_df = None
        self.address_features = None
        self.similarity_graph = None
        self.clusters = {}

    def load_data(self, tx_file="new_data/layerzero_transactions.csv"):
        try:
            self.transactions_df = pd.read_csv(tx_file)
            logging.info(f"📖 載入 {len(self.transactions_df)} 筆交易，{self.transactions_df['address'].nunique()} 個地址")
            self.transactions_df['block_timestamp'] = pd.to_datetime(self.transactions_df['block_timestamp'])
            self.transactions_df['created'] = pd.to_datetime(self.transactions_df['created'])
            return True
        except FileNotFoundError:
            logging.error(f"❌ 找不到交易檔案: {tx_file}")
            return False

    def extract_behavioral_features(self):
        logging.info("🔍 提取地址行為特徵...")
        features_list = []
        for address in self.transactions_df['address'].unique():
            addr_txs = self.transactions_df[self.transactions_df['address'] == address]
            features = {
                'address': address,
                'tx_count': len(addr_txs),
                'unique_src_chains': addr_txs['src_eid'].nunique(),
                'unique_dst_chains': addr_txs['dst_eid'].nunique(),
                'unique_pathways': addr_txs['pathway_id'].nunique(),
            }
            features_list.append(features)
        self.address_features = pd.DataFrame(features_list)
        return self.address_features

    def calculate_address_similarity(self):
        logging.info("🔗 計算地址間相似度...")
        numeric_features = self.address_features.select_dtypes(include=[np.number])
        features_scaled = StandardScaler().fit_transform(numeric_features.fillna(0))
        similarity_matrix = cosine_similarity(features_scaled)
        return similarity_matrix

    def detect_temporal_clusters(self, time_window_minutes=30):
        logging.info(f"⏰ 偵測時間集群（{time_window_minutes} 分鐘）...")
        temporal_clusters = []
        all_txs = self.transactions_df.sort_values('block_timestamp')
        current_cluster = []
        cluster_start_time = None
        for _, tx in all_txs.iterrows():
            tx_time = tx['block_timestamp']
            if cluster_start_time is None:
                cluster_start_time = tx_time
                current_cluster = [tx]
            elif (tx_time - cluster_start_time).total_seconds() <= time_window_minutes * 60:
                current_cluster.append(tx)
            else:
                if len(current_cluster) >= 2:
                    temporal_clusters.append({
                        'cluster_id': len(temporal_clusters),
                        'start_time': cluster_start_time,
                        'end_time': current_cluster[-1]['block_timestamp'],
                        'addresses': list(set(tx['address'] for tx in current_cluster)),
                        'transaction_count': len(current_cluster),
                        'unique_addresses': len(set(tx['address'] for tx in current_cluster))
                    })
                cluster_start_time = tx_time
                current_cluster = [tx]
        self.clusters['temporal'] = temporal_clusters
        return temporal_clusters

    def detect_pathway_clusters(self):
        logging.info("🛤️ 偵測路徑集群...")
        pathway_clusters = defaultdict(list)
        for address in self.transactions_df['address'].unique():
            addr_txs = self.transactions_df[self.transactions_df['address'] == address].sort_values('block_timestamp')
            pattern = "|".join(f"{tx['src_eid']}->{tx['dst_eid']}" for _, tx in addr_txs.iterrows())
            pathway_clusters[pattern].append(address)
        pathway_cluster_list = []
        for pattern, addresses in pathway_clusters.items():
            if len(addresses) >= 2:
                pathway_cluster_list.append({
                    'cluster_id': len(pathway_cluster_list),
                    'pattern': pattern,
                    'addresses': addresses,
                    'size': len(addresses)
                })
        self.clusters['pathway'] = pathway_cluster_list
        return pathway_cluster_list

    def detect_similarity_clusters(self, eps=0.1, min_samples=2):
        logging.info("🎯 使用 DBSCAN 檢測相似度集群...")
        numeric_features = self.address_features.select_dtypes(include=[np.number])
        features_scaled = StandardScaler().fit_transform(numeric_features.fillna(0))
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(features_scaled)
        clusters = []
        for cluster_id in set(labels):
            if cluster_id == -1: continue
            addrs = self.address_features[labels == cluster_id]['address'].tolist()
            clusters.append({
                'cluster_id': cluster_id,
                'addresses': addrs,
                'size': len(addrs)
            })
        self.clusters['similarity'] = clusters
        return clusters

    def generate_cluster_report(self):
        logging.info("📋 生成集群報告...")
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_addresses': self.address_features['address'].nunique(),
            'total_transactions': len(self.transactions_df),
            'clusters': {t: {'count': len(c), 'details': c} for t, c in self.clusters.items()}
        }
        with open("new_data/cluster_analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        logging.info("✅ 集群分析報告已儲存至 new_data/cluster_analysis_report.json")
        return report

def main():
    analyzer = SybilClusterAnalyzer()
    if not analyzer.load_data():
        return
    analyzer.extract_behavioral_features()
    analyzer.calculate_address_similarity()
    analyzer.detect_temporal_clusters()
    analyzer.detect_pathway_clusters()
    analyzer.detect_similarity_clusters()
    analyzer.generate_cluster_report()

if __name__ == "__main__":
    main()
