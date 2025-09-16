import pandas as pd
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_sybil_addresses():
    """清理重複地址並驗證格式"""
    
    # 讀取原始資料
    df = pd.read_csv("data/sybil_addresses.csv")
    original_count = len(df)
    logging.info(f"📖 讀取原始資料：{original_count} 個地址")
    
    # 1. 地址格式驗證
    valid_format = df['address'].str.match(r'^0x[a-fA-F0-9]{40}$')
    invalid_addresses = df[~valid_format]
    
    if len(invalid_addresses) > 0:
        logging.warning(f"❌ 發現 {len(invalid_addresses)} 個格式錯誤的地址")
        print("無效地址範例：")
        print(invalid_addresses.head())
    
    # 2. 移除重複地址（保留第一次出現）
    df_clean = df[valid_format].drop_duplicates(subset=['address'], keep='first')
    duplicates_removed = original_count - len(df_clean)
    
    logging.info(f"🔄 移除重複地址：{duplicates_removed} 個")
    logging.info(f"✅ 清理後地址數量：{len(df_clean)} 個")
    
    # 3. 加入地址統計資訊
    df_clean = df_clean.copy()
    df_clean['address_lower'] = df_clean['address'].str.lower()
    df_clean['created_at'] = pd.Timestamp.now()
    
    # 4. 儲存清理後的資料
    df_clean.to_csv("data/sybil_addresses_clean.csv", index=False)
    
    # 5. 產生清理報告
    report = {
        'original_count': original_count,
        'duplicates_removed': duplicates_removed,
        'invalid_format': len(invalid_addresses),
        'final_count': len(df_clean),
        'unique_sources': df_clean['source'].nunique()
    }
    
    # 儲存報告
    report_df = pd.DataFrame([report])
    report_df.to_csv("data/cleaning_report.csv", index=False)
    
    logging.info("📋 清理報告已儲存至 data/cleaning_report.csv")
    logging.info("🎯 清理完成！使用 sybil_addresses_clean.csv 進行後續分析")
    
    return df_clean, report

if __name__ == "__main__":
    clean_df, report = clean_sybil_addresses()
    
    print("\n" + "="*50)
    print("📊 清理結果摘要")
    print("="*50)
    print(f"原始地址數量：{report['original_count']:,}")
    print(f"重複地址移除：{report['duplicates_removed']:,}")
    print(f"格式錯誤移除：{report['invalid_format']:,}")
    print(f"最終地址數量：{report['final_count']:,}")
    print(f"資料品質提升：{((report['final_count']/report['original_count'])*100):.1f}% 保留")
    print("="*50)