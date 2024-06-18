# ADSB-X Data scraper

⚠️ This is a work in progress, and still much manual intervention for any modification & normal operation.

## Description

This project is about downloading and consolidating ADS-B type data from ADSB-Exchange. It has world coverage. Only one day of raw data is freely available.

## Data Cleaning Pipeline

The data cleaning pipeline includes the following steps:

- Dataset is downloaded and consolidated into one file per month. 
- Flights that are in the air when the day starts / end are excluded.
- Aircraft type is narrowed to B7* and A3* (most popular short-medium haul planes)
- Flights with missing longitudes & latitudes are ignored

## Steps: 

> Download pre-processed data from MinIO using minio_transfer.py --download

> Downloading raw data for modification:
>
>0. Setup runtime.yaml
>1. Call scraper eg for years 2021 to 2023 (python) python scraper.py --years 2021 2023
>2. Call consolidate.py to convert raw json data to parquet consolidated files
>3. Upload data to MinIo using python minio_transfer.py --upload


# ADS-B Data (Post - March 2020)

## Data Columns Description

- **hex**: ICAO 24-bit aircraft address in hexadecimal.
- **type**: Type of ADS-B message.
- **flight**: Flight identification (callsign).
- **r**: Receiver ID.
- **t**: Type of transmission.
- **gs**: Ground speed in knots.
- **track**: Aircraft's track angle over the ground.
- **geom_rate**: Geometric rate of climb or descent in feet per minute.
- **squawk**: Assigned transponder code.
- **emergency**: Emergency status indicator.
- **category**: Aircraft category (e.g., light, medium, heavy).
- **rr_lat**: Multilateration latitude.
- **rr_lon**: Multilateration longitude.
- **version**: ADS-B version number.
- **nac_v**: Navigation Accuracy Category for velocity.
- **sil_type**: Source Integrity Level type.
- **alert**: Alert flag indicating an emergency or alert condition.
- **spi**: Special Position Indicator flag.
- **mlat**: Indicates if the data is derived from multilateration.
- **tisb**: Indicates if the data is derived from Traffic Information Service-Broadcast.
- **messages**: Number of messages received from the aircraft.
- **seen**: Time since the last message was received in seconds.
- **rssi**: Received signal strength indicator.
- **alt_baro**: Barometric altitude in feet.
- **alt_geom**: Geometric altitude in feet.
- **baro_rate**: Barometric rate of climb or descent in feet per minute.
- **nav_qnh**: Altimeter setting in hectopascals (QNH).
- **nav_altitude_mcp**: Selected altitude from the Mode Control Panel.
- **nav_heading**: Selected heading.
- **lat**: Latitude in decimal degrees.
- **lon**: Longitude in decimal degrees.
- **nic**: Navigation Integrity Category.
- **rc**: Radius of Containment.
- **seen_pos**: Time since the last position update in seconds.
- **nic_baro**: Navigation Integrity Category for barometric altitude.
- **nac_p**: Navigation Accuracy Category for position.
- **sil**: Source Integrity Level.
- **gva**: Geometric Vertical Accuracy.
- **sda**: System Design Assurance.
- **nav_modes**: Navigation modes engaged.
- **ias**: Indicated airspeed in knots.
- **tas**: True airspeed in knots.
- **mach**: Mach number (speed relative to the speed of sound).
- **wd**: Wind direction in degrees.
- **ws**: Wind speed in knots.
- **oat**: Outside air temperature in degrees Celsius.
- **tat**: Total air temperature in degrees Celsius.
- **track_rate**: Rate of change of track angle in degrees per second.
- **roll**: Aircraft roll angle in degrees.
- **mag_heading**: Magnetic heading in degrees.
- **true_heading**: True heading in degrees.
- **dbFlags**: Database flags for internal use.
- **nav_altitude_fms**: Selected altitude from the Flight Management System.
- **calc_track**: Calculated track angle over the ground.


Specific values: 
- **alt_baro**: positive float64 and -1 for 'ground' values

# ADS-B Data (Pre - March 2020)

The ADS-B data includes the following fields:

- **Timestamp (now)**: Unix timestamp (float or integer).
- **ICAO Address (hex)**: Hexadecimal string representing the aircraft's unique ICAO address.
- **Callsign (flight)**: String representing the flight number or callsign.
- **Aircraft Type (type)**: String, usually 'adsb_icao' for ADS-B messages.
- **Registration (r)**: String representing the aircraft's registration number.
- **Aircraft Model (t)**: String representing the aircraft type or model.
- **Barometric Altitude (alt_baro)**: Either a float for altitude in feet or a special string like 'ground'.
- **Geometric Altitude (alt_geom)**: Float representing geometric altitude in feet.
- **Ground Speed (gs)**: Float representing ground speed in knots.
- **Track Angle (track)**: Float representing track angle in degrees.
- **Vertical Rate (baro_rate)**: Float representing the barometric altitude rate of change in feet per minute.
