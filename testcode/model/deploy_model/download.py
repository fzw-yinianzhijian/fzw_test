from huggingface_hub import snapshot_download
import os

download_id='google/gemma-7b-it'
download_path='./model_weights'

snapshot_download(repo_id=download_id, local_dir=download_path, token=os.getenv('huggingface_api'))

