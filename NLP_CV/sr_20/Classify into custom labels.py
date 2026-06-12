import kagglehub

# Download latest version
path = kagglehub.dataset_download("puskas78/eurlex-dataset")

print("Path to dataset files:", path)
