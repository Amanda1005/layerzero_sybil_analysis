# sybil_score.py
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SybilScorer:
    def __init__(self):
        self.transactions_df = None
        self.cluster_data = None
        self.address_scores = {}
        self.data_dir = "new_data"

        self.scoring_weights = {
            'pathway_similarity': 40,
            'temporal_coordination': 25,
            'behavioral_similarity': 20,
            'network_density': 15
        }

    def load_data(self):
        try:
            self.transactions_df = pd.read_csv(f"{self.data_dir}/layerzero_transactions.csv")
            self.transactions_df['block_timestamp'] = pd.to_datetime(self.transactions_df['block_timestamp'])
            logging.info(f"📖 載入交易數據: {len(self.transactions_df)} 筆")

            with open(f"{self.data_dir}/cluster_analysis_report.json", "r", encoding='utf-8') as f:
                self.cluster_data = json.load(f)
            logging.info("📖 載入集群分析結果")
            return True
        except FileNotFoundError as e:
            logging.error(f"❌ 找不到檔案: {str(e)}")
            return False

    def calculate_pathway_similarity_score(self, address):
        clusters = self.cluster_data.get('clusters', {}).get('pathway', {}).get('details', [])
        for cluster in clusters:
            if address in cluster.get('addresses', []):
                size = cluster.get('size', 0)
                return 100 if size >= 20 else 85 if size >= 10 else 70 if size >= 5 else 50
        return 10

    def calculate_temporal_coordination_score(self, address):
        clusters = self.cluster_data.get('clusters', {}).get('temporal', {}).get('details', [])
        max_score = 0
        for cluster in clusters:
            if address in cluster.get('addresses', []):
                u = cluster.get('unique_addresses', 1)
                t = cluster.get('transaction_count', 1)
                score = 100 if u >= 15 and t >= 100 else 85 if u >= 10 and t >= 50 else 70 if u >= 5 and t >= 20 else 40
                max_score = max(max_score, score)
        return max_score

    def calculate_behavioral_similarity_score(self, address):
        clusters = self.cluster_data.get('clusters', {}).get('similarity', {}).get('details', [])
        for cluster in clusters:
            if address in cluster.get('addresses', []):
                size = cluster.get('size', 0)
                return 90 if size >= 10 else 75 if size >= 5 else 60 if size >= 3 else 40
        return 20

    def calculate_network_density_score(self, address):
        df = self.transactions_df[self.transactions_df['address'] == address]
        if df.empty:
            return 0
        score = 0
        tx_count = len(df)
        unique_paths = df['pathway_id'].nunique()
        time_span = df['block_timestamp'].max() - df['block_timestamp'].min()
        score += 30 if tx_count == 8 else 20 if 6 <= tx_count <= 10 else 5
        score += 25 if unique_paths <= 2 else 15 if unique_paths <= 4 else 5
        if pd.notna(time_span):
            hours = time_span.total_seconds() / 3600
            score += 30 if hours <= 2 else 20 if hours <= 6 else 10
        return min(score, 100)

    def calculate_composite_score(self, address):
        scores = {
            'pathway_similarity': self.calculate_pathway_similarity_score(address),
            'temporal_coordination': self.calculate_temporal_coordination_score(address),
            'behavioral_similarity': self.calculate_behavioral_similarity_score(address),
            'network_density': self.calculate_network_density_score(address)
        }
        weighted = sum(scores[k] * (self.scoring_weights[k] / 100) for k in scores)
        return {
            'address': address,
            'composite_score': round(weighted, 2),
            'risk_level': self.get_risk_level(weighted),
            'component_scores': scores,
            'confidence': self.calculate_confidence(scores)
        }

    def get_risk_level(self, score):
        return "CRITICAL - 極高女巫風險" if score >= 90 else \
               "HIGH - 高女巫風險" if score >= 75 else \
               "MEDIUM - 中等女巫風險" if score >= 60 else \
               "LOW - 低女巫風險" if score >= 40 else \
               "MINIMAL - 最小風險"

    def calculate_confidence(self, scores):
        high = sum(1 for s in scores.values() if s >= 80)
        med = sum(1 for s in scores.values() if 60 <= s < 80)
        return "VERY_HIGH" if high >= 3 else "HIGH" if high >= 2 else "MEDIUM" if high >= 1 or med >= 3 else "LOW"

    def score_all_addresses(self):
        logging.info("🎯 開始計算所有地址的女巫風險分數...")
        scored = []
        for address in self.transactions_df['address'].unique():
            result = self.calculate_composite_score(address)
            scored.append(result)
            logging.info(f"✅ {address}: {result['composite_score']:.1f} ({result['risk_level']})")
        scored.sort(key=lambda x: x['composite_score'], reverse=True)
        return scored

    def generate_final_report(self, scored_addresses):
        logging.info("📋 生成最終女巫分析報告...")
        scores = [a['composite_score'] for a in scored_addresses]
        risk_levels = [a['risk_level'] for a in scored_addresses]
        report = {
            'analysis_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_addresses': len(scored_addresses),
                'total_transactions': len(self.transactions_df),
                'analysis_version': '1.0'
            },
            'summary_statistics': {
                'average_score': np.mean(scores),
                'median_score': np.median(scores),
                'max_score': max(scores),
                'min_score': min(scores),
                'std_score': np.std(scores)
            },
            'risk_distribution': {
                level: sum(level in r for r in risk_levels)
                for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
            },
            'detailed_scores': scored_addresses,
            'recommendations': self.generate_recommendations(scored_addresses)
        }

        os.makedirs(self.data_dir, exist_ok=True)

        # Save JSON
        with open(f"{self.data_dir}/final_sybil_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        # Save CSV
        summary_df = pd.DataFrame([
            {
                'address': a['address'],
                'composite_score': a['composite_score'],
                'risk_level': a['risk_level'],
                'confidence': a['confidence'],
                'pathway_score': a['component_scores']['pathway_similarity'],
                'temporal_score': a['component_scores']['temporal_coordination'],
                'behavioral_score': a['component_scores']['behavioral_similarity'],
                'network_score': a['component_scores']['network_density']
            } for a in scored_addresses
        ])
        summary_df.to_csv(f"{self.data_dir}/final_sybil_scores.csv", index=False)

        logging.info("💾 最終報告已儲存至 new_data/")
        return report

    def generate_recommendations(self, scored_addresses):
        crit = [a for a in scored_addresses if a['composite_score'] >= 90]
        high = [a for a in scored_addresses if 75 <= a['composite_score'] < 90]
        return [
            f"發現 {len(crit)} 個極高風險女巫地址，建議立即列入黑名單",
            f"發現 {len(high)} 個高風險地址，建議進一步調查",
            "所有地址顯示極度相似行為，建議建立自動監控",
            "建議對特定集群設置風險閾值"
        ]

    def print_summary_report(self, report):
        print("="*70)
        print("🕵️ LayerZero 女巫地址分析 - 最終報告")
        print("="*70)
        meta = report['analysis_metadata']
        stats = report['summary_statistics']
        print(f"📊 總地址數: {meta['total_addresses']}")
        print(f"📈 總交易數: {meta['total_transactions']}")
        print(f"📉 平均風險分數: {stats['average_score']:.1f}/100")
        print(f"🚨 最高風險分數: {stats['max_score']:.1f}/100\n")
        print("🚩 風險等級分佈:")
        for level, count in report['risk_distribution'].items():
            print(f"  {level}: {count}")
        print("🎯 建議:")
        for r in report['recommendations']:
            print(f"  - {r}")
        print("="*70)

def main():
    scorer = SybilScorer()
    if not scorer.load_data():
        return
    results = scorer.score_all_addresses()
    report = scorer.generate_final_report(results)
    scorer.print_summary_report(report)
    logging.info("✅ 女巫評分任務完成")

if __name__ == "__main__":
    main()
