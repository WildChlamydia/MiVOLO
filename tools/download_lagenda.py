import os
import zipfile

# pip install gdown
import gdown

if __name__ == "__main__":
    print("Download LAGENDA Age Gender Dataset... ")
    out_dir = "LAGENDA"
    os.makedirs(out_dir, exist_ok=True)

    ids = ["1QXO0NlkABPZT6x1_0Uc2i6KAtdcrpTbG", "1mNYjYFb3MuKg-OL1UISoYsKObMUllbJx"]
    dests = [f"{out_dir}/lagenda_benchmark_images.zip", f"{out_dir}/lagenda_annotation.csv"]

    for file_id, destination in zip(ids, dests):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, destination, quiet=False)

        if not os.path.exists(destination):
            print(f"ERROR: Can not download {destination}")
            continue

        if os.path.basename(destination).split(".")[-1] != ".zip":
            continue

        print(f"Extracting {destination} ... ")
        with zipfile.ZipFile(destination) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(f"./{out_dir}/")
        os.remove(destination)
