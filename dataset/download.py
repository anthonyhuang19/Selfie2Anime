import kagglehub

# Download latest version
path = kagglehub.dataset_download("arnaud58/selfie2anime")

print("Path to dataset files:", path)