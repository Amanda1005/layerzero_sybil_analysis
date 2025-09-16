#!/usr/bin/env python3
"""
檢查項目狀態和數據文件
"""

import os
import pandas as pd
import json
from pathlib import Path

def check_project_status():
    """檢查項目當前狀態"""
    print("🔍 LayerZero 女巫分析項目狀態檢查")
    print("=" * 50)
    
    # 檢查目錄結構
    print("\n📁 目錄結構:")
    base_dirs = ['data', 'scripts', 'notebooks', 'dashboard']
    for dir_name in base_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
            # 列出目錄內容
            for file in os.listdir(dir_name):
                file_path = os.path.join(dir_name, file)
                size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                print(f"   📄 {file} ({size} bytes)")
        else:
            print(f"❌ {dir_name}/ (不存在)")
    
    # 檢查關鍵數據文件
    print("\n📊 關鍵數據文件:")
    required_files = {
        'data/sybil_addresses.csv': '原始女巫地址列表',
        'data/layerzero_transactions.csv': '交易數據',
        'data/cluster_analysis_report.json': '集群分析結果',
        'data/final_sybil_scores.csv': '最終風險評分',
        'data/cluster_summary.csv': '集群摘要'
    }
    
    existing_files = {}
    missing_files = []
    
    for file_path, description in required_files.items():
        if os.path.exists(file_path):
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    rows = len(df)
                    cols = len(df.columns)
                    print(f"✅ {file_path}: {description} ({rows} 行, {cols} 列)")
                    existing_files[file_path] = {'rows': rows, 'cols': cols, 'type': 'csv'}
                elif file_path.endswith('.json'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    print(f"✅ {file_path}: {description} (JSON)")
                    existing_files[file_path] = {'type': 'json', 'size': len(str(data))}
            except Exception as e:
                print(f"⚠️  {file_path}: 存在但無法讀取 - {str(e)}")
        else:
            print(f"❌ {file_path}: {description} (缺失)")
            missing_files.append(file_path)
    
    # 檢查腳本文件
    print("\n🐍 Python 腳本:")
    script_files = [
        'scripts/fetch_tx_history.py',
        'scripts/cluster_analysis.py', 
        'scripts/sybil_score.py',
        'streamlit_sybil_dashboard.py'
    ]
    
    for script in script_files:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script}")
    
    # 分析當前狀態
    print("\n📋 狀態分析:")
    if 'data/sybil_addresses.csv' in existing_files:
        print("✅ 已有原始地址數據")
    else:
        print("❌ 缺少原始地址數據")
    
    if 'data/layerzero_transactions.csv' in existing_files:
        print("✅ 已有交易數據")
    else:
        print("❌ 缺少交易數據 - 需要運行 fetch_tx_history.py")
    
    if 'data/cluster_analysis_report.json' in existing_files:
        print("✅ 已有集群分析")
    else:
        print("❌ 缺少集群分析 - 需要運行 cluster_analysis.py")
    
    if 'data/final_sybil_scores.csv' in existing_files:
        print("✅ 已有風險評分")
    else:
        print("❌ 缺少風險評分 - 需要運行 sybil_score.py")
    
    # 提供修復建議
    print("\n💡 修復建議:")
    if missing_files:
        print("🔄 按順序執行以下命令來生成缺失的文件:")
        
        if 'data/layerzero_transactions.csv' in missing_files:
            print("1. python scripts/fetch_tx_history.py")
        
        if 'data/cluster_analysis_report.json' in missing_files:
            print("2. python scripts/cluster_analysis.py")
        
        if 'data/final_sybil_scores.csv' in missing_files:
            print("3. python scripts/sybil_score.py")
        
        print("4. streamlit run streamlit_sybil_dashboard.py")
    else:
        print("🎉 所有文件都存在，可以直接運行 Streamlit dashboard!")
        print("   streamlit run streamlit_sybil_dashboard.py")
    
    return existing_files, missing_files

def create_sample_data():
    """創建示例數據用於演示"""
    print("\n🎭 創建示例數據用於演示...")
    
    # 確保 data 目錄存在
    os.makedirs('data', exist_ok=True)
    
    # 如果沒有交易數據，創建示例數據
    if not os.path.exists('data/layerzero_transactions.csv'):
        print("📝 創建示例交易數據...")
        
        # 基於你之前的真實數據創建示例
        sample_data = []
        addresses = [f"0x{i:040x}" for i in range(20)]  # 20個示例地址
        
        import datetime
        base_time = datetime.datetime(2023, 7, 13, 20, 52, 13)
        
        for i, addr in enumerate(addresses):
            for j in range(8):  # 每個地址8筆交易
                sample_data.append({
                    'address': addr,
                    'guid': f'guid_{i}_{j}',
                    'src_tx_hash': f'0x{(i*8+j):064x}',
                    'dst_tx_hash': f'0x{(i*8+j+1000):064x}',
                    'src_eid': [102, 110, 106, 102, 109, 106, 110, 111][j],
                    'dst_eid': [116, 106, 102, 109, 106, 110, 111, 106][j],
                    'src_chain': 'EID_' + str([102, 110, 106, 102, 109, 106, 110, 111][j]),
                    'dst_chain': 'EID_' + str([116, 106, 102, 109, 106, 110, 111, 106][j]),
                    'nonce': j + 1,
                    'status': 'DELIVERED',
                    'source_status': 'DELIVERED',
                    'dest_status': 'DELIVERED',
                    'block_timestamp': base_time + datetime.timedelta(minutes=i*2+j*5),
                    'pathway_id': f'pathway_{j}'
                })
        
        df = pd.DataFrame(sample_data)
        df.to_csv('data/layerzero_transactions.csv', index=False)
        print("✅ 示例交易數據已創建")
    
    # 創建示例風險評分數據
    if not os.path.exists('data/final_sybil_scores.csv'):
        print("📝 創建示例風險評分數據...")
        
        addresses = [f"0x{i:040x}" for i in range(20)]
        scores_data = []
        
        for i, addr in enumerate(addresses):
            scores_data.append({
                'address': addr,
                'composite_score': 85 + (i % 5),  # 85-89分
                'risk_level': 'HIGH - 高女巫風險',
                'confidence': 'VERY_HIGH',
                'pathway_score': 100,
                'temporal_score': 85 + (i % 10),
                'behavioral_score': 80 + (i % 15),
                'network_score': 85 + (i % 8)
            })
        
        df_scores = pd.DataFrame(scores_data)
        df_scores.to_csv('data/final_sybil_scores.csv', index=False)
        print("✅ 示例風險評分數據已創建")
    
    # 創建示例集群分析數據
    if not os.path.exists('data/cluster_analysis_report.json'):
        print("📝 創建示例集群分析數據...")
        
        cluster_report = {
            "analysis_timestamp": "2025-09-11T11:46:39",
            "total_addresses": 20,
            "total_transactions": 160,
            "clusters": {
                "temporal": {
                    "count": 3,
                    "details": [
                        {
                            "cluster_id": 0,
                            "addresses": [f"0x{i:040x}" for i in range(20)],
                            "size": 20,
                            "transaction_count": 160
                        }
                    ]
                },
                "pathway": {
                    "count": 1,
                    "details": [
                        {
                            "cluster_id": 0,
                            "addresses": [f"0x{i:040x}" for i in range(20)],
                            "size": 20,
                            "pattern": "102->116|110->106|106->102|102->109|109->106|106->110|110->111|111->106"
                        }
                    ]
                },
                "similarity": {
                    "count": 3,
                    "details": [
                        {
                            "cluster_id": 0,
                            "addresses": [f"0x{i:040x}" for i in range(12)],
                            "size": 12
                        },
                        {
                            "cluster_id": 1,
                            "addresses": [f"0x{i:040x}" for i in range(12, 15)],
                            "size": 3
                        },
                        {
                            "cluster_id": 2,
                            "addresses": [f"0x{i:040x}" for i in range(15, 19)],
                            "size": 4
                        }
                    ]
                }
            }
        }
        
        with open('data/cluster_analysis_report.json', 'w', encoding='utf-8') as f:
            json.dump(cluster_report, f, indent=2, ensure_ascii=False)
        print("✅ 示例集群分析數據已創建")

if __name__ == "__main__":
    existing, missing = check_project_status()
    
    if missing:
        print(f"\n⚠️  發現 {len(missing)} 個缺失文件")
        
        response = input("\n❓ 是否創建示例數據用於演示? (y/n): ")
        if response.lower() == 'y':
            create_sample_data()
            print("\n🎉 示例數據創建完成！現在可以運行 Streamlit dashboard 了:")
            print("   streamlit run streamlit_sybil_dashboard.py")
        else:
            print("\n💡 請按照修復建議順序執行腳本來生成真實數據")
    else:
        print("\n🚀 一切就緒！運行以下命令啟動 dashboard:")
        print("   streamlit run streamlit_sybil_dashboard.py")