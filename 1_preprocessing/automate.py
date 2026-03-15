import os
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, '../data/raw/synthetic_fraud_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, '../data/processed')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'clean_card_transdata.csv')

def process_data():
    print("🚀 Memulai preprocessing data...")
    
    # 1. Load Data
    if not os.path.exists(RAW_DATA_PATH):
        print(f"File tidak ditemukan: {RAW_DATA_PATH}")
        return
    
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.dropna()

    # 2. Pemisahan Fitur dan Target (Sesuai draf kamu)
    # Catatan: device_risk_score & ip_risk_score dibuang sesuai drafmu
    target_col = 'is_fraud'
    drop_cols = [target_col, 'device_risk_score', 'ip_risk_score']
    
    X = df.drop(columns=drop_cols)
    y = df[target_col]

    # 3. Definisi Kolom
    num_cols = ['amount', 'hour']
    cat_cols = ['transaction_type', 'merchant_category', 'country']

    # 4. Transformasi (Scaling & Encoding)
    # Kita gunakan fit_transform pada seluruh data bersih untuk output dataset final
    scaler = StandardScaler()
    X_num = pd.DataFrame(scaler.fit_transform(X[num_cols]), columns=num_cols, index=X.index)

    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_cat_encoded = encoder.fit_transform(X[cat_cols])
    X_cat_df = pd.DataFrame(
        X_cat_encoded, 
        columns=encoder.get_feature_names_out(cat_cols), 
        index=X.index
    )

    # 5. Gabungkan Kembali
    df_final = pd.concat([X_num, X_cat_df, y], axis=1)

    # 6. Simpan Hasil
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df_final.to_csv(OUTPUT_FILE, index=False)
    print(f"Preprocessing selesai! Data disimpan di: {OUTPUT_FILE}")
    print(f"Ukuran data final: {df_final.shape}")

if __name__ == "__main__":
    process_data() 
    