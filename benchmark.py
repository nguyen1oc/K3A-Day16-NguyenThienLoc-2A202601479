import time
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb

# 1. Đo thời gian load data
print("Đang tải dataset...")
start_time = time.time()
# Đọc file creditcard.csv nằm cùng thư mục
df = pd.read_csv('creditcard.csv')
load_time = time.time() - start_time
print(f"Thời gian load data: {load_time:.4f} giây")

# 2. Phân chia tập dữ liệu train/test (80/20)
X = df.drop(columns=['Class'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Huấn luyện mô hình LightGBM
print("Đang huấn luyện mô hình LightGBM...")
train_start = time.time()
clf = lgb.LGBMClassifier(random_state=42, n_estimators=100)
clf.fit(
    X_train, 
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='binary_logloss',
    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
)
train_time = time.time() - train_start
print(f"Thời gian training: {train_time:.4f} giây")

best_iteration = clf.best_iteration_ if hasattr(clf, 'best_iteration_') else 100
print(f"Best iteration: {best_iteration}")

# 4. Đánh giá mô hình trên tập test
print("Đang đánh giá mô hình...")
y_pred_proba = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

auc_roc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"AUC-ROC: {auc_roc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# 5. Đo Inference Latency (Dự đoán cho 1 dòng)
# Thực hiện dự đoán tuần tự trên 100 dòng để lấy trung bình chính xác hơn
print("Đang đo Inference Latency (1 dòng)...")
latency_runs = []
for i in range(100):
    sample = X_test.iloc[[i]]
    start_inf = time.time()
    _ = clf.predict(sample)
    latency_runs.append(time.time() - start_inf)
avg_latency_ms = np.mean(latency_runs) * 1000
print(f"Inference latency (1 row): {avg_latency_ms:.4f} ms")

# 6. Đo Inference Throughput (Dự đoán cho 1000 dòng đồng thời)
print("Đang đo Inference Throughput (1000 dòng)...")
throughput_samples = X_test.head(1000)
start_thru = time.time()
_ = clf.predict(throughput_samples)
thru_time = time.time() - start_thru
throughput_fps = 1000 / thru_time
print(f"Inference throughput (1000 rows): {throughput_fps:.2f} rows/second (Thời gian thực hiện: {thru_time:.4f} giây)")

# 7. Ghi kết quả ra file benchmark_result.json
result = {
    "data_loading_time_seconds": load_time,
    "training_time_seconds": train_time,
    "best_iteration": best_iteration,
    "auc_roc": auc_roc,
    "accuracy": accuracy,
    "f1_score": f1,
    "precision": precision,
    "recall": recall,
    "inference_latency_1_row_ms": avg_latency_ms,
    "inference_throughput_1000_rows_per_second": throughput_fps
}

output_path = 'benchmark_result.json'
with open(output_path, 'w') as f:
    json.dump(result, f, indent=4)
print(f"Đã lưu kết quả thành công vào file: {output_path}")
