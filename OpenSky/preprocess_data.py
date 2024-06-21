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
import argparse

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


def plot_training_data(training_data_path: str) -> None:

    plt.style.use("ggplot")

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": EuroPP()})
    training_data = Traffic.from_file(training_data_path)
    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )

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
    plt.savefig(
        f"{save_path}/opensky_training_data_{ADEP_code}_to_{ADES_code}.png", bbox_inches="tight"
    )
    print(f"Saved figure to {save_path}/opensky_training_data_{ADEP_code}_to_{ADES_code}.png")



def plot_training_data_with_altitude(training_data_path: str) -> None:
    # Set up the map
    training_data = Traffic.from_file(training_data_path)
    ADEP_code, ADES_code, geographic_extent = extract_geographic_info(
        training_data_path
    )
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
    plt.savefig(
        f"{save_path}/opensky_training_data_with_altitude_{ADEP_code}_to_{ADES_code}.png",
        bbox_inches="tight",
    )
    print(f"Saved figure to {save_path}/opensky_training_data_with_altitude_{ADEP_code}_to_{ADES_code}.png")



def get_trajectories(flights_points: pd.DataFrame) -> Traffic:

    # Convert timestamp to datetime object
    flights_points["timestamp"] = pd.to_datetime(
        flights_points["timestamp"], format="%d-%m-%Y %H:%M:%S", utc=True
    )

    # Create Flight objects for each unique flight ID
    flights_list = [
        Flight(flights_points[flights_points["flight_id"] == flight_id])
        for flight_id in flights_points["flight_id"].unique()
    ]

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
        
    latitude_threshold, longitude_threshold, altitude_threshold, lowest_sequence_length_threshold = thresholds
    latitude_outliers = find_outliers_zscore(opensky_data, 'latitude', threshold=latitude_threshold)
    print(f"Found {len(latitude_outliers)} outliers in column 'latitude', with threshold {latitude_threshold}")
    print(latitude_outliers[['flight_id', 'latitude']])
    print(latitude_outliers['flight_id'].unique())
    print(f"Number of unique flight ids: {latitude_outliers['flight_id'].nunique()}")

    longitude_outliers = find_outliers_zscore(opensky_data, 'longitude', threshold=longitude_threshold)
    print(f"Found {len(longitude_outliers)} outliers in column 'longitude', with threshold {longitude_threshold}")
    print(longitude_outliers[['flight_id', 'longitude']])
    print(longitude_outliers['flight_id'].unique())
    print(f"Number of unique flight ids: {longitude_outliers['flight_id'].nunique()}")

    altitude_outliers = find_outliers_zscore(opensky_data, 'altitude', threshold=altitude_threshold)
    print(f"Found {len(altitude_outliers)} outliers in column 'altitude', with threshold {altitude_threshold}")
    print(altitude_outliers[['flight_id', 'altitude']])
    print(altitude_outliers['flight_id'].unique())
    print(f"Number of unique flight ids: {altitude_outliers['flight_id'].nunique()}")

    # drop rows with altitude outliers
    opensky_data = opensky_data.drop(altitude_outliers.index).reset_index(drop=True)
    # drop flights with latitude outliers
    opensky_data = opensky_data[~opensky_data['flight_id'].isin(latitude_outliers['flight_id'])]

    # drop flights with longitude outliers if they are not already dropped
    longitude_outliers = longitude_outliers[~longitude_outliers['flight_id'].isin(latitude_outliers['flight_id'])]
    opensky_data = opensky_data[~opensky_data['flight_id'].isin(longitude_outliers['flight_id'])]

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
    print(low_counts_outliers)

    # drop the low counts outliers
    opensky_data = opensky_data[~opensky_data['flight_id'].isin(low_counts_outliers['flight_id'])]
    # reset the index
    opensky_data = opensky_data.reset_index(drop=True)

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


    trajectories = get_trajectories(opensky_data)

    trajectories = prepare_trajectories(
        trajectories, int(avg_sequence_length), n_jobs=7, douglas_peucker_coeff=None
    )

    save_path = "./opensky_traffic.pkl"
    trajectories.to_pickle(save_path)

    # Plot the training data
    plot_training_data(training_data_path=save_path)

    plot_training_data_with_altitude(training_data_path=save_path)

    # remove the saved file
    os.remove(save_path)


if __name__ == "__main__":

    # use argesparse to get the arguments: threshold values for outliers, window for flight id assignment, file path to opensky data
    parser = argparse.ArgumentParser(description="Preprocess OpenSky data")
    parser.add_argument(
        "--thresholds",
        nargs=3,
        type=float,
        default=[2.0, 3.0, 2.2, -1.4],
        help="Threshold values for outliers in latitude, longitude, and altitude, and lowest sequence length",
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
        default="./opensky_EHAM_LIMC_2019-10-01_2019-12-01.csv",
        help="Path to OpenSky data",
    )
    args = parser.parse_args()
    main(args)

