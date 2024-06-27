from traffic.core import Flight, Traffic
import os
from typing import Any, Dict, List, Tuple
import cartopy.feature
import pandas as pd
import seaborn as sns
from cartopy.crs import EuroPP, PlateCarree
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
from traffic.data import airports
from scipy.stats import zscore
from math import radians, sin, cos, sqrt, atan2
import argparse


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance


def calculate_consecutive_distances(df, distance_threshold):
    """Calculates distances between consecutive points and flags flights with any excessive distance."""
    # Calculate distances for each point to the next within each flight
    df = df.sort_values(['flight_id', 'timestamp'])
    df['next_latitude'] = df.groupby('flight_id')['latitude'].shift(-1)
    df['next_longitude'] = df.groupby('flight_id')['longitude'].shift(-1)

    # Apply the Haversine formula
    df['segment_distance'] = df.apply(
        lambda row: haversine(row['latitude'], row['longitude'], row['next_latitude'], row['next_longitude']) 
        if not pd.isna(row['next_latitude']) else 0, axis=1
    )

    # Find flights with any segment exceeding the threshold
    outlier_flights = df[df['segment_distance'] > distance_threshold]['flight_id'].unique()
    return outlier_flights 


def calculate_initial_distance(df, origin_lat_lon, distance_threshold):
    """Calculates distances between the first point in each flight and the origin airport."""
    # Calculate distances from the origin airport to the first point of each flight

    # first point of each flight
    first_points = df.groupby('flight_id').first()
    # Calculate distances from the origin airport to the first point of each flight
    first_points['initial_distance'] = [haversine(lat, lon, origin_lat_lon[0], origin_lat_lon[1]) for lat, lon in zip(first_points['latitude'], first_points['longitude'])]

    # Find flights with the first point exceeding the threshold
    outlier_flights = first_points[first_points['initial_distance'] > distance_threshold].index
    return outlier_flights

def calculate_final_distance(df, destination_lat_lon, distance_threshold):
    """Calculates distances between the last point in each flight and the destination airport."""
    # Calculate distances from the destination airport to the last point of each flight

    # last point of each flight
    last_points = df.groupby('flight_id').last()
    # Calculate distances from the destination airport to the last point of each flight
    last_points['final_distance'] = [haversine(lat, lon, destination_lat_lon[0], destination_lat_lon[1]) for lat, lon in zip(last_points['latitude'], last_points['longitude'])]

    # Find flights with the last point exceeding the threshold
    outlier_flights = last_points[last_points['final_distance'] > distance_threshold].index
    return outlier_flights



def extract_geographic_info(
    training_data_path: str,
) -> Tuple[str, str, float, float, float, float, float, float]:
    from traffic.core import Traffic

    training_data = Traffic.from_file(training_data_path)

    # raise an error if there exists more than one destination airport or if there are more than one origin airport
    if len(training_data.data["ADES"].unique()) > 1:
        raise ValueError("There are multiple destination airports in the training data")
    if len(training_data.data["ADES"].unique()) == 0:
        raise ValueError("There are no destination airports in the training data")
    if len(training_data.data["ADEP"].unique()) > 1:
        raise ValueError("There are multiple origin airports in the training data")
    if len(training_data.data["ADEP"].unique()) == 0:
        raise ValueError("There are no origin airports in the training data")

    ADEP_code = training_data.data["ADEP"].value_counts().idxmax()
    ADES_code = training_data.data["ADES"].value_counts().idxmax()

    # Determine the geographic bounds for plotting
    lon_min = training_data.data["longitude"].min()
    lon_max = training_data.data["longitude"].max()
    lat_min = training_data.data["latitude"].min()
    lat_max = training_data.data["latitude"].max()

    # Padding to apply around the bounds
    lon_padding = 1
    lat_padding = 1

    geographic_extent = [
        lon_min - lon_padding,
        lon_max + lon_padding,
        lat_min - lat_padding,
        lat_max + lat_padding,
    ]

    return ADEP_code, ADES_code, geographic_extent


def plot_training_data(training_data_path: str, ADEP_code: str, ADES_code: str, geographic_extent: List[float]) -> None:

    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": EuroPP()})
    training_data = Traffic.from_file(training_data_path)

    training_data.plot(ax, alpha=0.2, color="darkblue", linewidth=1)

    ax.coastlines()
    ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1.0)
    ax.set_extent(geographic_extent)

    # Plot the origin and destination airports
    airports[ADEP_code].point.plot(
        ax, color="red", label=f"Origin: {ADEP_code}", s=500, zorder=5
    )
    airports[ADES_code].point.plot(
        ax, color="green", label=f"Destination: {ADES_code}", s=500, zorder=5
    )

    plt.title(f"Training data of flight trajectories from {ADEP_code} to {ADES_code}")
    plt.legend(loc="upper right")

    # Add gridlines
    ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")

    # tight layout
    plt.tight_layout()

    # save the figure to the figures directory in the data directory
    data_directory = os.path.dirname(training_data_path)
    save_path = os.path.join(data_directory, "figures")
    os.makedirs(save_path, exist_ok=True)
    # save_name as training_data_path and replace the file extension with .png
    save_name = os.path.basename(training_data_path).replace(".pkl", ".png")
    plt.savefig(
        f"{save_path}/{save_name}", bbox_inches="tight"
    )
    print(f"Saved figure to {save_path}/{save_name}")

    # plt.savefig(
    #     f"{save_path}/opensky_training_data_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight"
    # )
    # print(f"Saved figure to {save_path}/opensky_training_data_{ADEP_code}_to_{ADES_code}.png")



def plot_training_data_with_altitude(training_data_path: str, ADEP_code: str, ADES_code: str, geographic_extent: List[float]) -> None:
    # Set up the map
    training_data = Traffic.from_file(training_data_path)
    df = training_data.data
    fig, ax = plt.subplots(figsize=(13, 12))
    m = Basemap(
        projection="merc",
        llcrnrlat=geographic_extent[2],
        urcrnrlat=geographic_extent[3],
        llcrnrlon=geographic_extent[0],
        urcrnrlon=geographic_extent[1],
        lat_ts=20,
        resolution="i",
    )

    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color="lightgray", lake_color="aqua")
    m.drawmapboundary(fill_color="aqua")

    # Convert latitude and longitude to x and y coordinates
    x, y = m(df["longitude"].values, df["latitude"].values)

    # Connect points with a line
    plt.plot(x, y, color="black", alpha=0.2, zorder=1)

    # Plot the points with altitude as hue and size
    sns.scatterplot(
        x=x,
        y=y,
        hue=df["altitude"],
        palette="viridis",
        size=df["altitude"],
        sizes=(20, 200),
        legend="brief",
        ax=ax,
        edgecolor="black",
        alpha=0.1
    )

    # Add color bar for altitude
    norm = plt.Normalize(vmin=df["altitude"].min(), vmax=df["altitude"].max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array(
        []
    )  # This is necessary because of a Matplotlib bug when using scatter with norm.
    cbar = plt.colorbar(sm, ax=ax, aspect=30)
    cbar.set_label("Altitude")
    # set legend upper right
    plt.legend(loc="upper right")

    # Add title and labels
    plt.title(f"Flight Path from {ADEP_code} to {ADES_code}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Adjust layout and display plot
    plt.tight_layout()

    data_directory = os.path.dirname(training_data_path)
    save_path = os.path.join(data_directory, "figures")
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.basename(training_data_path).replace(".pkl", "_with_altitude.png")
    plt.savefig(
        f"{save_path}/{save_name}", bbox_inches="tight"
    )
    print(f"Saved figure to {save_path}/{save_name}")
    # plt.savefig(
    #     f"{save_path}/opensky_training_data_with_altitude_{ADEP_code}_to_{ADES_code}.png",
    #     bbox_inches="tight",
    # )
    # print(f"Saved figure to {save_path}/opensky_training_data_with_altitude_{ADEP_code}_to_{ADES_code}.png")



def get_trajectories(flights_points: pd.DataFrame) -> Traffic:

    # Convert timestamp to datetime object
    flights_points["timestamp"] = pd.to_datetime(
        flights_points["timestamp"], format="%d-%m-%Y %H:%M:%S", utc=True
    )

    # Create Flight objects for each unique flight ID
    # flights_list = [
    #     Flight(flights_points[flights_points["flight_id"] == flight_id])
    #     for flight_id in flights_points["flight_id"].unique()
    # ]
    # Group the DataFrame by 'flight_id' and then create Flight objects
    grouped_flights = flights_points.groupby('flight_id')
    flights_list = [Flight(group) for _, group in grouped_flights]


    # Create a Traffic object containing all the flights
    trajectories = Traffic.from_flights(flights_list)
    return trajectories


def prepare_trajectories(
    trajectories: Traffic, n_samples: int, n_jobs: int, douglas_peucker_coeff: float
) -> Traffic:
    # Resample trajectories for uniformity
    trajectories = (
        trajectories.resample(n_samples)
        .unwrap()
        .eval(max_workers=n_jobs, desc="resampling")
    )
    trajectories = trajectories.compute_xy(projection=EuroPP())

    # Simplify trajectories with Douglas-Peucker algorithm if a coefficient is provided
    if douglas_peucker_coeff is not None:
        print("Simplification...")
        trajectories = trajectories.simplify(tolerance=1e3).eval(desc="")

    # Add elapsed time since start for each flight
    trajectories = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in trajectories
    )

    return trajectories



def assign_flight_ids(opensky_data: pd.DataFrame, window: int = 6) -> pd.DataFrame:

    # Initialize the flight_id column and a dictionary to track the last time per (icao24, callsign) df['flight_id'] = None
    opensky_data['flight_id'] = None
    last_flight_times = {}

    # Function to determine flight id based on past flight times and a 6-hour window
    def assign_flight_id_fn(row, window=window):
        key = (row['icao24'], row['callsign'])
        current_time = row['timestamp']
        if key in last_flight_times and (current_time - last_flight_times[key]['time']).total_seconds() / 3600 <= window:
            # If within 6 hours of the last flight with the same icao24 and callsign, use the same flight id
            return last_flight_times[key]['flight_id']
        else:
            # Otherwise, create a new flight id
            formatted_time = current_time.strftime('%Y%m%d_%H%M%S')
            # new_flight_id = f"{row['icao24']}_{row['callsign']}_{current_time.isoformat()}"
            new_flight_id = f"{row['icao24']}_{row['callsign']}_{formatted_time}"
            last_flight_times[key] = {'time': current_time, 'flight_id': new_flight_id}
            return new_flight_id


    # Apply the function to each row df['flight_id'] = df.apply(assign_flight_id, axis=1)
    opensky_data['flight_id'] = opensky_data.apply(assign_flight_id_fn, axis=1)
    
    return opensky_data

def remove_outliers(opensky_data: pd.DataFrame, thresholds: List[float]) -> Tuple[pd.DataFrame, float]:

    # print the number of unique flight ids
    num_flights = opensky_data['flight_id'].nunique()
    print(f"Number of unique flight ids before removing outliers: {num_flights}")



    def find_outliers_zscore(df, column, threshold=2.5):
        # Calculate z-scores
        df['z_score'] = zscore(df[column])
        
        # Filter and return outlier rows
        outliers = df[df['z_score'].abs() > threshold]
        return outliers.drop(columns='z_score')
        
    consecutive_distance_threshold, altitude_threshold, lowest_sequence_length_threshold = thresholds


    consecutive_distance_outliers = calculate_consecutive_distances(opensky_data, distance_threshold=consecutive_distance_threshold)
    print(f"Found {len(consecutive_distance_outliers)} flights with excessive consecutive distances.")

    ADEP_code = opensky_data['ADEP'].value_counts().idxmax()
    ADES_code = opensky_data['ADES'].value_counts().idxmax()
    ADEP_lat_lon = airports[ADEP_code].latlon
    ADES_lat_lon = airports[ADES_code].latlon
    # find outliers where the distance between the first point in the flight and the origin airport is greater than 100 km
    initial_distance_outliers = calculate_initial_distance(opensky_data, ADEP_lat_lon, distance_threshold=100)
    print(f"Found {len(initial_distance_outliers)} flights with excessive initial distances.")
    print(f"Number of unique flight ids in initial distance outliers that are in consecutive distance outliers: {len(set(initial_distance_outliers).intersection(set(consecutive_distance_outliers)))}")

    # find outliers where the distance between the last point in the flight and the destination airport is greater than 100 km
    final_distance_outliers = calculate_final_distance(opensky_data, ADES_lat_lon, distance_threshold=100)
    print(f"Found {len(final_distance_outliers)} flights with excessive final distances.")
    print(f"Number of unique flight ids in final distance outliers that are in consecutive distance outliers: {len(set(final_distance_outliers).intersection(set(consecutive_distance_outliers)))}")
    print(f"Number of unique flight ids in final distance outliers that are in initial distance outliers: {len(set(final_distance_outliers).intersection(set(initial_distance_outliers)))}")

    altitude_outliers = find_outliers_zscore(opensky_data, 'altitude', threshold=altitude_threshold)
    print(f"Found {len(altitude_outliers)} outliers in column 'altitude', with threshold {altitude_threshold}")
    print(altitude_outliers[['flight_id', 'altitude']])
    # print(altitude_outliers['flight_id'].unique())
    print(f"Number of unique flight ids in altitude outliers: {altitude_outliers['flight_id'].nunique()}\n")



    # drop rows with altitude outliers
    print("Dropping rows with altitude outliers...")
    opensky_data = opensky_data.drop(altitude_outliers.index).reset_index(drop=True)
 
    # drop flights with consecutive distance outliers
    print("Dropping flights with consecutive distance outliers...")
    opensky_data = opensky_data[~opensky_data['flight_id'].isin(consecutive_distance_outliers)]

    # drop flights with initial distance outliers that are not dropped by consecutive distance outliers
    initial_distance_outliers = [flight_id for flight_id in initial_distance_outliers if flight_id not in consecutive_distance_outliers]
    print("Dropping flights with initial distance outliers...")
    opensky_data = opensky_data[~opensky_data['flight_id'].isin(initial_distance_outliers)]

    # drop flights with final distance outliers that are not dropped by consecutive distance outliers or initial distance outliers
    final_distance_outliers = [flight_id for flight_id in final_distance_outliers if flight_id not in consecutive_distance_outliers and flight_id not in initial_distance_outliers]
    print("Dropping flights with final distance outliers...")
    opensky_data = opensky_data[~opensky_data['flight_id'].isin(final_distance_outliers)]

    # reset the index
    opensky_data = opensky_data.reset_index(drop=True)
     
    # find the average number of rows in each flight with unique flight_id
    avg_sequence_length = opensky_data.groupby("flight_id").size().mean()
    
    # count the number of rows in each flight with unique flight_id, and make it a dataframe
    size = opensky_data.groupby("flight_id").size().reset_index(name='counts')

    # calculate z-scores for the counts
    size['z_score'] = zscore(size['counts'])

    # drop flights with lowest sequence length
    low_counts_outliers = size[size['z_score'] < lowest_sequence_length_threshold]
    print(f"Found {len(low_counts_outliers)} outliers in column 'counts', with threshold {lowest_sequence_length_threshold}")
    # print(low_counts_outliers)

    # drop the low counts outliers
    opensky_data = opensky_data[~opensky_data['flight_id'].isin(low_counts_outliers['flight_id'])]
    # reset the index
    opensky_data = opensky_data.reset_index(drop=True)

    # remove flights with duplicate rows of the same timestamp. To address: ValueError: cannot reindex on an axis with duplicate labels
    duplicate_rows = opensky_data[opensky_data.duplicated(subset=['flight_id', 'timestamp'], keep=False)]
    duplicate_flights = duplicate_rows['flight_id'].unique()
    print(f"Found {len(duplicate_flights)} flights with duplicate rows")
    opensky_data = opensky_data[~opensky_data['flight_id'].isin(duplicate_flights)]

    return opensky_data, avg_sequence_length


def main(args):


    # opensky_data = pd.read_csv("./pensky_EHAM_LIMC_2019-10-01_2019-12-01.csv")
    opensky_data = pd.read_csv(args.opensky_data)
    # drop Unnamed: 0 column
    opensky_data = opensky_data.drop(columns=['Unnamed: 0'])
    # opensky_data = opensky_data.drop(columns=['groundspeed', 'track', 'geoaltitude'])

    # drop the rows with Nan values and reset the index
    opensky_data = opensky_data.dropna().reset_index(drop=True)

    # change the column names to match the ectrl data: estdepartureairport	estarrivalairport, to ADEP and ADES
    opensky_data = opensky_data.rename(columns={"estdepartureairport": "ADEP", "estarrivalairport": "ADES"})

    # Convert the 'timestamp' column to datetime
    opensky_data['timestamp'] = pd.to_datetime(opensky_data['timestamp'])
    #df.sort_values('timestamp', inplace=True)
    opensky_data.sort_values('timestamp', inplace=True)

    # assign flight ids
    opensky_data = assign_flight_ids(opensky_data, window=args.window)

    # remove outliers
    opensky_data, avg_sequence_length = remove_outliers(opensky_data, args.thresholds)

    print("Removed outliers, now getting trajectories...")
    trajectories = get_trajectories(opensky_data)

    print("Preparing trajectories...")
    trajectories = prepare_trajectories(
        trajectories, int(avg_sequence_length), n_jobs=7, douglas_peucker_coeff=None
    )

    # save_path = "./opensky_traffic.pkl"
    # replace the file extension with .pkl
    save_path = args.opensky_data.replace(".csv", ".pkl")
    trajectories.to_pickle(save_path)

    # Plot the training data
    print("Plotting training data...")

    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path=save_path
    )
    plot_training_data(training_data_path=save_path, ADEP_code=ADEP_code, ADES_code=ADES_code, geographic_extent=geographic_extent)

    # print("Plotting training data with altitude...")
    plot_training_data_with_altitude(training_data_path=save_path, ADEP_code=ADEP_code, ADES_code=ADES_code, geographic_extent=geographic_extent)

    # remove the saved file
    # os.remove(save_path)


if __name__ == "__main__":

    # use argesparse to get the arguments: threshold values for outliers, window for flight id assignment, file path to opensky data
    parser = argparse.ArgumentParser(description="Preprocess OpenSky data")
    parser.add_argument(
        "--thresholds",
        nargs=3,
        type=float,
        default=[50, 2.2, -1.4],
        help="Threshold values for consecutive distance, altitude, and lowest sequence length",
    )


    parser.add_argument(
        "--window",
        type=int,
        default=6,
        help="Window in hours for flight id assignment",
    )
    parser.add_argument(
        "--opensky_data",
        type=str,
        default="./opensky_EHAM_LIMC_2019-01-01_2020-01-01.csv",
        help="Path to OpenSky data",
    )
    args = parser.parse_args()
    main(args)

