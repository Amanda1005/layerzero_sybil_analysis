# 🕵️ LayerZero Sybil Detection Analysis

This project analyzes Sybil behaviors on LayerZero by fetching transaction data, performing behavioral clustering, and scoring risk levels of each wallet address.


## Project Highlights

- **Analyzed 10,000 addresses**
- **Fetched 380,000+ transactions** from LayerZero API
- **Scored Sybil risk** using pathway similarity, time coordination, behavior, and network density
- **Interactive dashboard** for HR/PMs to explore risk levels, trends, and top pathways

## Dashboard Preview

Run with:

```bash
streamlit run streamlit_dashboard.py

Sample Analysis Result:

Total Addresses: 10,000

Total Transactions: 380,112

Avg Risk Score: 55.9 / 100

Max Risk Score: 94.25 / 100

Installation
pip install -r requirements.txt

Powered By:

Python, Pandas, Plotly, Streamlit

LayerZero Scan API

Custom clustering + scoring engine