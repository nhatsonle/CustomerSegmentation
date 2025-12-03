# Customer Segmentation System

        A modular, extensible customer segmentation system that analyzes customer purchasing behavior and creates actionable customer segments. Built using Python, Streamlit, and scikit-learn.

        ## Features

        - **Data Loading & Preprocessing**: Clean and preprocess raw retail transaction data
        - **Feature Engineering**: Create meaningful customer-level features
        - **Customer Clustering**: Segment customers using KMeans clustering
        - **Segment Interpretation**: Automatically interpret clusters and create customer personas
        - **Dynamic Segmentation**: Update segments as new data becomes available
        - **Interactive Visualization**: Explore segments through an intuitive Streamlit dashboard
        - **Marketing Recommendations**: Get actionable marketing strategies for each segment

        ## Project Structure

        ```
        .
        ├── data/                 # Data directory
        │   ├── raw/              # Raw data files
        │   └── processed/        # Processed data files
        ├── models/               # Saved models
        ├── reports/              # Reports and visualizations
        │   └── figures/          # Generated figures
        ├── src/                  # Source code
        │   ├── data/             # Data loading and preprocessing
        │   ├── features/         # Feature engineering
        │   ├── models/           # Clustering models
        │   ├── visualization/    # Visualization and dashboard
        │   └── utils/            # Helper utilities
        ├── requirements.txt      # Project dependencies
        ├── README.md             # Project documentation
        └── run.py                # Script to run the application
        ```
        