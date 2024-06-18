import spin_tools as st

st.export_environment_variables(runtime_env_filename='runtime.yaml')

directory_uploader = st.S3DirectoryUploader(bucket_name='synthair', object_name='ADSB-X', local_path='data/consolidated')

directory_downloader = st.S3DirectoryDownloader(bucket_name='synthair', object_name='ADSB-X', local_path='data/ADSB-X')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Transfer data to and from Minio')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--upload', action='store_true', help='Upload data to Minio')
    group.add_argument('--download', action='store_true', help='Download data from Minio')
    args = parser.parse_args()

    if args.upload:
        directory_uploader.upload()
    elif args.download:
        directory_downloader.download()