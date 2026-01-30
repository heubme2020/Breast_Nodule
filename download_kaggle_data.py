import kagglehub

# Download latest version
path = kagglehub.dataset_download("tommyngx/inbreast2012")

print("Path to dataset files:", path)