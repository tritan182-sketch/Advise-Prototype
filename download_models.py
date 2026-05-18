import urllib.request
import os

def download_file(url, destination):
    print(f"Downloading {destination} from {url}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response, open(destination, 'wb') as out_file:
            out_file.write(response.read())
        print(f"[SUCCESS] Saved to {os.path.abspath(destination)}")
    except Exception as e:
        print(f"[ERROR] Download failed for {destination}: {e}")

if __name__ == "__main__":
    # Corrected complete explicit raw network file paths
    proto_url = "https://githubusercontent.com"
    model_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel"
    
    # Skip redownloading the prototxt file if it already exists safely
    if not os.path.exists("deploy.prototxt"):
        download_file(proto_url, "deploy.prototxt")
        
    download_file(model_url, "mobilenet_iter_73000.caffemodel")