import os
import requests
from tqdm import tqdm
import concurrent.futures
import time
import argparse

# Define the base URL
base_url = "https://samples.adsbexchange.com/readsb-hist/"

force_download = False

# Define the range of years, months, days, hours, and minutes
months = range(1, 13)
days = range(1, 2)
hours = range(0, 24)
minutes = range(0, 60)



def download_file(year, month, day, hour, minute, pbar, force_download=False):
    url = f"{base_url}{year}/{str(month).zfill(2)}/{str(day).zfill(2)}/{str(hour).zfill(2)}{str(minute).zfill(2)}00Z.json.gz"
    local_dirname = f"data/{url.replace(base_url, '')}"
    if not os.path.exists(local_dirname) or force_download:
        response = requests.head(url)
        if response.status_code == 200:
            response = requests.get(url)
            file_data = response.content
            os.makedirs(os.path.dirname(local_dirname), exist_ok=True)
            with open(local_dirname, 'wb') as f:
                f.write(file_data)
            pbar.update(1)
    time.sleep(0.002)

def download_dataset(years, force_download=False, workers=None):

    print('Downloading files for years:', years)
    # Print a colourfull warning saying that the progress bar might not be representative of the actual progress
    print("\033[93mWarning: The progress bar might not be representative of the actual progress.\033[0m")


    total_files = len(years) * len(months) * len(days) * len(hours) * len(minutes)
    with tqdm(total=total_files, desc="Downloading files") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            for year in years:
                for month in months:
                    for day in days:
                        for hour in hours:
                            for minute in minutes:
                                futures.append(executor.submit(download_file, year, month, day, hour, minute, pbar, force_download))
            for future in concurrent.futures.as_completed(futures):
                pass
    
    # Print a nice download compelted message
    print("\033[92mDownload completed!\033[0m")
    

def get_parser():
    parser = argparse.ArgumentParser(description="Download files for specified years.")
    parser.add_argument("--years", nargs="+", type=int, required=True, help="Space-separated list of years to download files for.")
    parser.add_argument("--force", action="store_true", required=False, default=False, help="Force download of files.")
    parser.add_argument("--workers", type=int, required=False, default=100, help="Number of workers to use for downloading files.")
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Convert the list of years to a range
    years = range(min(args.years), max(args.years) + 1)
    download_dataset(years, force_download=args.force, workers=args.workers)



if __name__ == "__main__":
    main()