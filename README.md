# Customer Clustering Project

## 🎯 Tổng Quan Dự Án

Dự án phân cụm khách hàng (Customer Clustering) sử dụng 3 thuật toán machine learning:
- **Hierarchical Clustering** (Ward, Complete, Average, Single linkage)
- **K-means Clustering** (với tối ưu hóa số cụm)
- **DBSCAN** (density-based clustering)

**Dataset:** UK Online Retail - 4,372 khách hàng với 16 đặc trưng đã được xử lý

---

## 📊 Các Thuật Toán Được Triển Khai

**03_clustering.ipynb** hiện tại chứa:
1. **K-Means Clustering** - Partitioning-based, xác định số cụm optimal bằng Elbow Method và Silhouette Score
2. **DBSCAN** - Density-based, tự động phát hiện cụm và điểm lân cận  
3. **Hierarchical Clustering** - Agglomerative, hỗ trợ 4 phương pháp liên kết (Ward, Complete, Average, Single)

**So sánh và đánh giá** được thực hiện trong **05_validation.ipynb**

---

## 📁 Cấu Trúc Project

```
CustomerSegmentation/CustumerCluster/
├── configs/
│   └── config.yaml              # Cấu hình tham số clustering
├── data/
│   └── transformed/             # Dữ liệu đã xử lý
│       ├── customer_features_transformed.csv
│       └── customer_features_scaled.csv
├── notebooks/
│   ├── 01_cleaning_and_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_clustering.ipynb      # K-Means, DBSCAN, Hierarchical
│   ├── 04_business_insights.ipynb
│   └── 05_validation.ipynb
├── src/
│   ├── clustering.py            # Thực hiện 3 thuật toán clustering
│   ├── clustering_library.py    # Thư viện clustering mở rộng
│   ├── data_loader.py           # Load và chuẩn bị dữ liệu
│   ├── evaluation.py            # Đánh giá clustering
│   ├── preprocessing.py         # Tiền xử lý dữ liệu
│   └── visualization.py         # Trực quan hóa kết quả
├── results/
│   ├── figures/                 # Biểu đồ và hình ảnh
│   ├── reports/                 # Báo cáo chi tiết
│   ├── cluster_assignments.csv  # Kết quả clustering
│   └── algorithm_comparison.csv # So sánh thuật toán
├── main.py                      # Script chính để chạy phân tích
├── requirements.txt             # Dependencies
├── .gitignore                   # Git ignore file
└── README.md                    # File này
```

---

## 🚀 Hướng Dẫn Chạy Project

### Bước 1: Cài Đặt Dependencies

```bash
# Di chuyển vào thư mục project
cd d:\CustomerSegmentation\CustumerCluster

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### Bước 2: Chạy Phân Tích Clustering

#### **Option 1: Chạy Script Python (Khuyến Nghị)**

```bash
python main.py
```

**Script này sẽ:**
- ✅ Load dữ liệu từ `data/transformed/customer_features_transformed.csv`
- ✅ Chạy 3 thuật toán clustering (K-means, DBSCAN, Hierarchical)
- ✅ Tạo visualizations và lưu vào `results/figures/`
- ✅ Tính toán metrics và lưu vào `results/reports/`
- ✅ Lưu kết quả cuối cùng vào `results/final_clustering_results.csv`

**Kết quả sẽ được lưu tại:**
```
results/
├── figures/
│   ├── kmeans_metrics.png              # K-Means elbow & silhouette
│   ├── dbscan_analysis.png             # DBSCAN parameter grid
│   ├── hierarchical_analysis.png       # Hierarchical linkage methods
│   ├── dendrogram.png                  # Dendrogram visualization
│   ├── algorithm_comparison.png        # So sánh 3 thuật toán
│   ├── clustering_comparison_pca.png   # PCA visualization
│   ├── cluster_distribution.png        # Phân bố cụm
│   ├── cluster_profiles.png            # Heatmap đặc trưng
│   ├── stability_analysis.png          # Kiểm tra ổn định
│   ├── feature_sensitivity.png         # Độ nhạy đặc trưng
│   ├── outlier_impact.png              # Tác động của outlier
│   └── algorithm_metrics_comparison.png # So sánh metrics
├── reports/
│   ├── algorithm_comparison.csv        # Bảng so sánh
│   ├── business_insights_summary.txt   # Tóm tắt kinh doanh
│   └── validation_report.txt           # Báo cáo xác nhận
├── cluster_assignments.csv             # Kết quả clustering tất cả
└── final_clustering_results.csv        # Backward compatibility
```

#### **Option 2: Sử dụng Jupyter Notebook**

```bash
# Khởi động Jupyter
jupyter notebook

# Mở file notebooks/03_clustering.ipynb
# Chạy từng cell để xem phân tích chi tiết
```

---

## ✅ Checklist Trước Khi Chạy

- [ ] Python 3.8+ đã cài đặt
- [ ] Dependencies đã cài: `pip install -r requirements.txt`
- [ ] Dữ liệu tồn tại trong `data/processed/`
- [ ] Config file đã được tạo (hoặc dùng default)
- [ ] Có quyền ghi vào folder `results/`

