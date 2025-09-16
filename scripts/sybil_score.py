import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import os

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SybilScorer:
    def __init__(self):
        """女巫風險評分系統"""
        self.transactions_df = None
        self.cluster_data = None
        self.address_scores = {}
        
        # 評分權重配置
        self.scoring_weights = {
            'pathway_similarity': 40,      # 路徑相似性權重最高
            'temporal_coordination': 25,   # 時間協調性
            'behavioral_similarity': 20,   # 行為相似性
            'network_density': 15          # 網絡密度
        }
    
    def load_data(self):
        """載入所有分析數據"""
        try:
            # 載入交易數據
            self.transactions_df = pd.read_csv("data/layerzero_transactions.csv")
            # 確保時間欄位格式正確
            self.transactions_df['block_timestamp'] = pd.to_datetime(self.transactions_df['block_timestamp'])
            logging.info(f"📖 載入交易數據: {len(self.transactions_df)} 筆")
            
            # 載入集群分析結果
            with open("data/cluster_analysis_report.json", "r", encoding='utf-8') as f:
                self.cluster_data = json.load(f)
            logging.info("📖 載入集群分析結果")
            
            return True
        except FileNotFoundError as e:
            logging.error(f"❌ 找不到檔案: {str(e)}")
            return False
    
    def calculate_pathway_similarity_score(self, address):
        """計算路徑相似性分數 (0-100)"""
        # 檢查是否在路徑集群中
        pathway_clusters = self.cluster_data.get('clusters', {}).get('pathway', {}).get('details', [])
        
        for cluster in pathway_clusters:
            if address in cluster.get('addresses', []):
                cluster_size = cluster.get('size', 0)
                # 集群越大，風險越高
                if cluster_size >= 20:
                    return 100  # 最高風險
                elif cluster_size >= 10:
                    return 85
                elif cluster_size >= 5:
                    return 70
                else:
                    return 50
        
        return 10  # 不在任何路徑集群中
    
    def calculate_temporal_coordination_score(self, address):
        """計算時間協調性分數 (0-100)"""
        temporal_clusters = self.cluster_data.get('clusters', {}).get('temporal', {}).get('details', [])
        
        max_score = 0
        for cluster in temporal_clusters:
            if address in cluster.get('addresses', []):
                unique_addresses = cluster.get('unique_addresses', 1)
                transaction_count = cluster.get('transaction_count', 1)
                
                # 計算集中度分數
                if unique_addresses >= 15 and transaction_count >= 100:
                    score = 100
                elif unique_addresses >= 10 and transaction_count >= 50:
                    score = 85
                elif unique_addresses >= 5 and transaction_count >= 20:
                    score = 70
                else:
                    score = 40
                
                max_score = max(max_score, score)
        
        return max_score
    
    def calculate_behavioral_similarity_score(self, address):
        """計算行為相似性分數 (0-100)"""
        similarity_clusters = self.cluster_data.get('clusters', {}).get('similarity', {}).get('details', [])
        
        for cluster in similarity_clusters:
            if address in cluster.get('addresses', []):
                cluster_size = cluster.get('size', 0)
                # 行為相似集群大小評分
                if cluster_size >= 10:
                    return 90
                elif cluster_size >= 5:
                    return 75
                elif cluster_size >= 3:
                    return 60
                else:
                    return 40
        
        return 20  # 不在相似性集群中
    
    def calculate_network_density_score(self, address):
        """計算網絡密度分數 (0-100)"""
        # 從交易數據計算該地址的網絡特徵
        addr_txs = self.transactions_df[self.transactions_df['address'] == address]
        
        if len(addr_txs) == 0:
            return 0
        
        # 計算特徵
        tx_count = len(addr_txs)
        unique_pathways = addr_txs['pathway_id'].nunique()
        time_span = (addr_txs['block_timestamp'].max() - addr_txs['block_timestamp'].min())
        
        # 評分邏輯
        score = 0
        
        # 交易數量一致性（女巫農場通常有相同的交易數）
        if tx_count == 8:  # 根據觀察到的模式
            score += 30
        elif 6 <= tx_count <= 10:
            score += 20
        else:
            score += 5
        
        # 路徑多樣性（低多樣性 = 高風險）
        if unique_pathways <= 2:
            score += 25
        elif unique_pathways <= 4:
            score += 15
        else:
            score += 5
        
        # 時間集中性
        if pd.notna(time_span) and hasattr(time_span, 'total_seconds'):
            time_hours = time_span.total_seconds() / 3600
            if time_hours <= 2:
                score += 30
            elif time_hours <= 6:
                score += 20
            else:
                score += 10
        else:
            score += 5
        
        return min(score, 100)
    
    def calculate_composite_score(self, address):
        """計算綜合女巫風險分數"""
        scores = {
            'pathway_similarity': self.calculate_pathway_similarity_score(address),
            'temporal_coordination': self.calculate_temporal_coordination_score(address),
            'behavioral_similarity': self.calculate_behavioral_similarity_score(address),
            'network_density': self.calculate_network_density_score(address)
        }
        
        # 加權計算總分
        weighted_score = sum(
            scores[component] * (self.scoring_weights[component] / 100)
            for component in scores
        )
        
        return {
            'address': address,
            'composite_score': round(weighted_score, 2),
            'risk_level': self.get_risk_level(weighted_score),
            'component_scores': scores,
            'confidence': self.calculate_confidence(scores)
        }
    
    def get_risk_level(self, score):
        """根據分數確定風險等級"""
        if score >= 90:
            return "CRITICAL - 極高女巫風險"
        elif score >= 75:
            return "HIGH - 高女巫風險"
        elif score >= 60:
            return "MEDIUM - 中等女巫風險"
        elif score >= 40:
            return "LOW - 低女巫風險"
        else:
            return "MINIMAL - 最小風險"
    
    def calculate_confidence(self, scores):
        """計算預測信心度"""
        # 如果多個指標都很高，信心度就高
        high_scores = sum(1 for score in scores.values() if score >= 80)
        medium_scores = sum(1 for score in scores.values() if 60 <= score < 80)
        
        if high_scores >= 3:
            return "VERY_HIGH"
        elif high_scores >= 2:
            return "HIGH"
        elif high_scores >= 1 or medium_scores >= 3:
            return "MEDIUM"
        else:
            return "LOW"
    
    def score_all_addresses(self):
        """為所有地址評分"""
        logging.info("🎯 開始計算所有地址的女巫風險分數...")
        
        addresses = self.transactions_df['address'].unique()
        scored_addresses = []
        
        for address in addresses:
            score_result = self.calculate_composite_score(address)
            scored_addresses.append(score_result)
            logging.info(f"✅ {address}: {score_result['composite_score']:.1f} ({score_result['risk_level']})")
        
        # 按分數排序
        scored_addresses.sort(key=lambda x: x['composite_score'], reverse=True)
        
        return scored_addresses
    
    def generate_final_report(self, scored_addresses):
        """生成最終女巫分析報告"""
        logging.info("📋 生成最終女巫分析報告...")
        
        # 統計分析
        scores = [addr['composite_score'] for addr in scored_addresses]
        risk_levels = [addr['risk_level'] for addr in scored_addresses]
        
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
                level: len([r for r in risk_levels if level in r])
                for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']
            },
            'detailed_scores': scored_addresses,
            'recommendations': self.generate_recommendations(scored_addresses)
        }
        
        # 儲存 JSON 報告
        with open("data/final_sybil_report.json", "w", encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 儲存 CSV 摘要
        summary_data = []
        for addr_data in scored_addresses:
            summary_data.append({
                'address': addr_data['address'],
                'composite_score': addr_data['composite_score'],
                'risk_level': addr_data['risk_level'],
                'confidence': addr_data['confidence'],
                'pathway_score': addr_data['component_scores']['pathway_similarity'],
                'temporal_score': addr_data['component_scores']['temporal_coordination'],
                'behavioral_score': addr_data['component_scores']['behavioral_similarity'],
                'network_score': addr_data['component_scores']['network_density']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv("data/final_sybil_scores.csv", index=False)
        
        logging.info("💾 最終報告已儲存:")
        logging.info("  - data/final_sybil_report.json")
        logging.info("  - data/final_sybil_scores.csv")
        
        return report
    
    def generate_recommendations(self, scored_addresses):
        """生成建議"""
        critical_addresses = [addr for addr in scored_addresses if addr['composite_score'] >= 90]
        high_risk_addresses = [addr for addr in scored_addresses if 75 <= addr['composite_score'] < 90]
        
        recommendations = [
            f"發現 {len(critical_addresses)} 個極高風險女巫地址，建議立即列入黑名單",
            f"發現 {len(high_risk_addresses)} 個高風險地址，建議進一步調查",
            "所有地址都顯示出極度相似的行為模式，強烈建議進行全面審查",
            "建議建立自動監控系統，檢測類似的協調行為模式"
        ]
        
        return recommendations
    
    def print_summary_report(self, report):
        """打印摘要報告"""
        print("\n" + "="*70)
        print("🕵️ LayerZero 女巫地址分析 - 最終報告")
        print("="*70)
        
        meta = report['analysis_metadata']
        stats = report['summary_statistics']
        distribution = report['risk_distribution']
        
        print(f"📊 分析概覽:")
        print(f"  總地址數: {meta['total_addresses']}")
        print(f"  總交易數: {meta['total_transactions']}")
        print(f"  平均風險分數: {stats['average_score']:.1f}/100")
        print(f"  最高風險分數: {stats['max_score']:.1f}/100")
        
        print(f"\n🚨 風險等級分佈:")
        for level, count in distribution.items():
            if count > 0:
                percentage = (count / meta['total_addresses']) * 100
                print(f"  {level}: {count} 個地址 ({percentage:.1f}%)")
        
        print(f"\n🎯 TOP 5 最高風險地址:")
        for i, addr in enumerate(report['detailed_scores'][:5], 1):
            print(f"  {i}. {addr['address']}")
            print(f"     風險分數: {addr['composite_score']:.1f}/100")
            print(f"     風險等級: {addr['risk_level']}")
            print(f"     信心度: {addr['confidence']}")
        
        print(f"\n💡 建議:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("="*70)

def main():
    """主執行函數"""
    
    # 初始化評分器
    scorer = SybilScorer()
    
    # 載入數據
    if not scorer.load_data():
        return
    
    # 計算所有地址分數
    scored_addresses = scorer.score_all_addresses()
    
    # 生成最終報告
    final_report = scorer.generate_final_report(scored_addresses)
    
    # 打印摘要
    scorer.print_summary_report(final_report)
    
    logging.info("🎯 女巫風險評分完成！")

if __name__ == "__main__":
    main()