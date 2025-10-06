import pandas as pd
import requests
import time
import logging
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LayerZeroTxFetcher:
    def __init__(self, rate_limit=3):
        """
        LayerZero 交易歷史抓取器
        rate_limit: 每秒最大請求數
        """
        self.rate_limit = rate_limit
        self.base_url = "https://scan.layerzero-api.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'LayerZero-Sybil-Analysis/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        # LayerZero 主要鏈 ID 映射 (EID - Endpoint ID)
        self.eid_mapping = {
            30101: "Ethereum",
            30102: "BNB Chain", 
            30106: "Avalanche",
            30109: "Polygon",
            30110: "Arbitrum",
            30111: "Optimism",
            30112: "Fantom",
            30183: "Linea",
            30184: "Base",
            30125: "Celo",
        }
    
    def get_wallet_transactions(self, address, limit=100):
        """
        使用正確的 LayerZero Scan API 抓取錢包交易
        """
        try:
            # 使用 wallet API endpoint
            url = f"{self.base_url}/messages/wallet/{address.lower()}"
            
            params = {
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=30)
            time.sleep(1/self.rate_limit)  # 速率限制
            
            if response.status_code == 200:
                data = response.json()
                return self._parse_wallet_response(address, data)
            elif response.status_code == 404:
                # 地址沒有 LayerZero 交易記錄
                logging.info(f"📭 {address}: 沒有 LayerZero 交易記錄")
                return []
            else:
                logging.warning(f"API 錯誤 {response.status_code} for {address}")
                return []
                
        except Exception as e:
            logging.error(f"抓取 {address} 交易失敗: {str(e)}")
            return []
    
    def _parse_wallet_response(self, address, data):
        """解析 LayerZero API 錢包回應"""
        transactions = []
        
        if 'data' in data and data['data']:
            for message in data['data']:
                try:
                    # 解析基本資訊
                    pathway = message.get('pathway', {})
                    source = message.get('source', {})
                    destination = message.get('destination', {})
                    source_tx = source.get('tx', {})
                    dest_tx = destination.get('tx', {})
                    
                    transaction = {
                        'address': address.lower(),
                        'guid': message.get('guid', ''),
                        'src_tx_hash': source_tx.get('txHash', ''),
                        'dst_tx_hash': dest_tx.get('txHash', ''),
                        'src_eid': pathway.get('srcEid'),
                        'dst_eid': pathway.get('dstEid'),
                        'src_chain': self.eid_mapping.get(pathway.get('srcEid'), f"EID_{pathway.get('srcEid')}"),
                        'dst_chain': self.eid_mapping.get(pathway.get('dstEid'), f"EID_{pathway.get('dstEid')}"),
                        'nonce': pathway.get('nonce'),
                        'status': message.get('status', {}).get('name', ''),
                        'source_status': source.get('status', ''),
                        'dest_status': destination.get('status', ''),
                        'block_timestamp': source_tx.get('blockTimestamp'),
                        'block_number': source_tx.get('blockNumber'),
                        'from_address': source_tx.get('from', ''),
                        'value': source_tx.get('value', '0'),
                        'created': message.get('created', ''),
                        'updated': message.get('updated', ''),
                        'sender_address': pathway.get('sender', {}).get('address', ''),
                        'receiver_address': pathway.get('receiver', {}).get('address', ''),
                        'pathway_id': pathway.get('id', '')
                    }
                    transactions.append(transaction)
                except Exception as e:
                    logging.warning(f"解析交易失敗: {str(e)}")
                    continue
        
        return transactions
    
    def fetch_batch_transactions(self, addresses, batch_size=20, max_workers=3):
        """
        批量抓取多個地址的交易歷史
        減少併發數和批次大小以避免 API 限制
        """
        all_transactions = []
        addresses_with_txs = 0
        
        # 分批處理
        for i in range(0, len(addresses), batch_size):
            batch = addresses[i:i+batch_size]
            logging.info(f"處理批次 {i//batch_size + 1}/{(len(addresses)+batch_size-1)//batch_size}, 地址數量: {len(batch)}")
            
            # 多線程處理（減少併發數）
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_address = {
                    executor.submit(self.get_wallet_transactions, addr): addr 
                    for addr in batch
                }
                
                for future in as_completed(future_to_address):
                    address = future_to_address[future]
                    try:
                        transactions = future.result()
                        if transactions:
                            all_transactions.extend(transactions)
                            addresses_with_txs += 1
                            logging.info(f"✅ {address}: {len(transactions)} 筆交易")
                        else:
                            logging.info(f"📭 {address}: 無交易記錄")
                    except Exception as e:
                        logging.error(f"❌ {address}: {str(e)}")
            
            # 每批次間暫停更長時間
            time.sleep(3)
        
        logging.info(f"🎯 總結: {addresses_with_txs}/{len(addresses)} 個地址有 LayerZero 交易")
        return all_transactions
    
    def save_transactions(self, transactions, filename="data/layerzero_transactions.csv"):
        """儲存交易資料"""
        if not transactions:
            logging.warning("沒有交易資料可以儲存")
            return None
        
        df = pd.DataFrame(transactions)
        
        # 資料清理和格式化
        if 'block_timestamp' in df.columns:
            df['block_timestamp'] = pd.to_datetime(df['block_timestamp'], unit='s', errors='coerce')
        
        if 'created' in df.columns:
            df['created'] = pd.to_datetime(df['created'], errors='coerce')
        
        if 'updated' in df.columns:
            df['updated'] = pd.to_datetime(df['updated'], errors='coerce')
        
        # 儲存 CSV
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        
        logging.info(f"💾 {len(df)} 筆交易已儲存至 {filename}")
        
        return df
    
    def generate_summary_stats(self, df):
        """生成統計摘要"""
        if df is None or len(df) == 0:
            return {}
        
        stats = {
            'total_transactions': len(df),
            'unique_addresses': df['address'].nunique(),
            'unique_chains': df[['src_chain', 'dst_chain']].stack().nunique(),
            'status_distribution': df['status'].value_counts().to_dict(),
            'chain_pairs': df.groupby(['src_chain', 'dst_chain']).size().to_dict(),
            'date_range': {
                'earliest': df['block_timestamp'].min(),
                'latest': df['block_timestamp'].max()
            } if 'block_timestamp' in df.columns else None,
            'top_pathways': df['pathway_id'].value_counts().head(5).to_dict()
        }
        
        return stats

def main():
    """主執行函數"""

    # 設定儲存路徑
    output_folder = "new_data"
    os.makedirs(output_folder, exist_ok=True)

    # 配置分析規模
    ANALYSIS_CONFIG = {
        'address_count': 10000,
        'batch_size': 20,
        'max_workers': 3,
        'rate_limit': 3
    }

    # 1. 讀取清理後的地址清單
    try:
        df_addresses = pd.read_csv("data/sybil_addresses_clean.csv")
        addresses = df_addresses['address'].tolist()
        logging.info(f"📖 載入 {len(addresses)} 個地址")
    except FileNotFoundError:
        logging.error("❌ 找不到地址清單")
        return

    # 2. 初始化抓取器
    fetcher = LayerZeroTxFetcher(rate_limit=ANALYSIS_CONFIG['rate_limit'])

    # 3. 選擇分析地址數量
    test_addresses = addresses[:ANALYSIS_CONFIG['address_count']]

    logging.info(f"🎯 分析模式：{len(test_addresses)} 個地址")
    estimated_minutes = len(test_addresses) * 2.5 / 60
    logging.info(f"⏱️  預估時間：{estimated_minutes:.1f} 分鐘")

    # 4. 抓取交易資料
    transactions = fetcher.fetch_batch_transactions(
        test_addresses, 
        batch_size=ANALYSIS_CONFIG['batch_size'],
        max_workers=ANALYSIS_CONFIG['max_workers']
    )

    # 5. 儲存結果
    if transactions:
        tx_path = os.path.join(output_folder, "layerzero_transactions.csv")
        df_tx = fetcher.save_transactions(transactions, filename=tx_path)

        stats = fetcher.generate_summary_stats(df_tx)

        stats_path = os.path.join(output_folder, "fetch_stats.csv")
        stats_df = pd.DataFrame([{
            'timestamp': datetime.now(),
            'addresses_tested': len(test_addresses),
            'addresses_with_txs': stats.get('unique_addresses', 0),
            'total_transactions': stats.get('total_transactions', 0),
            'success_rate': f"{(stats.get('unique_addresses', 0)/len(test_addresses)*100):.1f}%"
        }])
        stats_df.to_csv(stats_path, index=False)

        logging.info(f"📋 統計報告已儲存至 {stats_path}")
    else:
        logging.warning("⚠️ 無成功抓取任何交易資料")

        
        # 提供下一步建議
        print("\n" + "="*60)
        print("💡 建議的替代方案:")
        print("="*60)
        print("1. 這些地址可能沒有 LayerZero 跨鏈活動")
        print("2. 可以考慮使用其他 API (如 Etherscan) 獲取一般交易")
        print("3. 檢查地址是否參與其他協議的活動")
        print("4. 增加測試地址數量或檢查不同時間段")
        print("="*60)

if __name__ == "__main__":
    main()