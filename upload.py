import os
from argparse import ArgumentParser
from google.cloud import storage

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
        )
    )


def download_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.download_to_filename(source_file_name)

    print(
        "File {} downloaded to {}.".format(
            destination_blob_name, source_file_name
        )
    )

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--local_path", type=str, default="2D_separable_points/")
    parser.add_argument("--remote_path", type=str, default="pe_test_Final/")

    args = parser.parse_args()

    for file in os.listdir(args.local_path):        
        upload_blob("aiml-lucey-research-data", os.path.join(args.local_path,file), "jianqiao/"+os.path.join(args.remote_path,args.local_path,file))