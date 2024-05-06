# Datasets repository: an overview of the available datasets for the SynthAir Project

One location with summary about datasets explored as part of the SynthAir project

## Summary table

- ✅ Data received
- 👍 Samples received
- 🟠 Samples available
- ❌ Samples unavailable

Name | Type | Size | Region | Price | Note
--- | --- | --- | --- | --- | ---
[Eurocontrol Flights](#eurocontrol-data) | Flight Plan | xx MB | Euro | Free | ✅
[Eurocontrol Points](#eurocontrol-data) | Flight Plan | xx MB | Euro | Free | ✅
[Eurocontrol FIRs](#eurocontrol-data) | Flight Regions | xx MB | Euro | Free | ✅
[Eurocontrol AUAs](#eurocontrol-data) | Flight Regions | xx MB | Euro | Free | ✅
[Eurocontrol Route](#eurocontrol-data) | Routes Info | xx MB | World | Free | ✅
[ADSB-X samples](#ads-b-exchange) | ADS-B | xx MB | World | Public | 👍 1st day /month
[Brazil-Flights](#brazil-flights) | Flight Table | 83 MB -> 13.3+ GB | Brazil | Public | ✅ ⚠️ Source unclear
[OTS](#ots-data) | Flight Table | xx MB | US | Public | ✅ 
[PlaneFinder](#planefinder) | ADS-B | xx MB | World | $$$ | 🟠 Contacted for samples
[PlaneFinder](#planefinder) | Flight Table | xx MB | World | $$$ | 🟠 Contacted for samples
[AvDelphi](#avdelphi) | Flight Table | xx MB | World | $$$ | ❌ Paid API
[AvDelphi](#avdelphi) | NOTAM | xx MB | World | $$$ | ❌ Paid API
[AvDelphi](#avdelphi) | ACARS | xx MB | World | $$$ | ❌ Web access only
[ACI World](#aci-world) | Airport: volumes | xx MB | World | 6600$ | 👍 For 2022 and 2021
[OpenSky](#opensky) | ADS-B | xx MB | World | Public | ❌ & ✅ Sebastian has access
[OpenSky](#opensky) | Live-API | xx MB | World | Public | 🟠
[OpenSky](#opensky) | Misc | xx MB | World | Public | 🟠
[OpenSky](#opensky) | Voice records | xx MB | World | Public | ⚙️ Work in Progress
[DataWorld](#data-world) | Misc | xx MB | Misc | Public | 🟠 Small volumes
[KiltHub TrajAir](#kilthub) | ADS-B | ~400 MB | USA | Public | ✅ General Aviation only


## Eurocontrol data

The data from Eurocontrol consists of 28 zip files, each corresponding to a specific month and year. The files are named as `YYYYMM.zip`, where `YYYY` is the year and `MM` is the month. The data spans from 2015 to 2021, but only four months (March, June, September, and December) are available for each year.

In each zip file, there are several CSV files. The most important files are:

- `Flights_YYYYMMDD_YYYYMMDD.csv`: contains the flight data for the specific month. See the [Flights.ipynb](/EuroControl/Flights.ipynb) notebook for more details.
- `Flight_Points_[Filed/Actual]_YYYYMMDD_YYYYMMDD.csv`: contains the points crossed by the flights in chronological order. See the [flight_Points.ipynb](/EuroControl/flight_points.ipynb) notebook for more details.
- `Flight_FIRs_[Filed/Actual]_YYYYMMDD_YYYYMMDD.csv`: contains the FIRs (Flight Information Regions) crossed by the flights in chronological order. See the [flight_information_regions.ipynb](/EuroControl/flight_information_region.ipynb) notebook for more details.
  - `FIR_YYMM.csv`: contains information about FIR.
- `Flight_AUAs_[Filed/Actual]_YYYYMMDD_YYYYMMDD.csv`: contains the AUAs (ATC Unit airspace) crossed by the flights in chronological order. See the [AUAs.ipynb](/EuroControl/AUAs.ipynb) notebook for more details.
- `Route_YYMM.csv`: contains information about traffic routes. See the [routes.ipynb](/EuroControl/routes.ipynb) and [route_discrepancy_study.ipynb](/EuroControl/route_discrepancy_study.ipynb) notebook for more details.

⬆️ [To summary table](#summary-table)

## ADS-B Exchange

The data from ADS-B Exchange is available free of charge for the first 24 hours of every month starting 2016.

Roughly 3-4 Gb of data per year

Sample data is available [here](https://www.adsbexchange.com/products/historical-data/)

- **readsb-hist**: Snapshots of all global airborne traffic are archived every 5 seconds starting May 2020, (prior data is available every 60 secs from starting in July 2016). You can access this data at the following URL, replacing `yyyy`, `mm`, and `dd` with the year, month, and day respectively: [https://samples.adsbexchange.com/readsb-hist/yyyy/mm/dd](https://samples.adsbexchange.com/readsb-hist/yyyy/mm/dd)

- **Trace Files**: Activity by individual ICAO hex for all aircraft during one 24-hour period are sub-organized by last two digits of hex code. You can access this data at the following URL, replacing `yyyy`, `mm`, `dd`, and `xx` with the year, month, day, and last two digits of the hex code respectively: [https://samples.adsbexchange.com/traces/yyyy/mm/dd/xx/](https://samples.adsbexchange.com/traces/yyyy/mm/dd/xx/)

- **hires-traces**: Same as trace files, but with an even higher sample rate of 2x per second, for detailed analysis of flightpaths, accidents, etc. You can access this data at the following URL, replacing `yyyy`, `mm`, `dd`, and `xx` with the year, month, day, and last two digits of the hex code respectively: [https://samples.adsbexchange.com/hires-traces/yyyy/mm/dd/xx](https://samples.adsbexchange.com/hires-traces/yyyy/mm/dd/xx)

- **ACAS**: TCAS/ACAS alerts detected by our ground stations, by day. You can access this data at the following URL, replacing `yyyy`, `mm`, and `dd` with the year, month, and day respectively: [https://samples.adsbexchange.com/acas/yyyy/mm/dd/](https://samples.adsbexchange.com/acas/yyyy/mm/dd/)

⬆️ [To summary table](#summary-table)

## Brazil Flights

Alledgedly from the [National Civil Aviation Agency](https://www.gov.br/anac/en). Can be downloaded from [kaggle](https://www.kaggle.com/datasets/ramirobentes/flights-in-brazil), but direct provenance should be prefered (unavailable for now)

A public domain collection of flight schedules recorded by the Brazilian civil air administration over the course of 10 years, available for download [here](https://www.kaggle.com/datasets/ramirobentes/flights-in-brazil/data), and basic vizualisations are located [here](/Brazil/brazil.ipynb)

⚠️ Approx.time appears approximate. Requires further investigation
⚠️ Delay in minutes must be infered from departure time and elapsed time vs crs_elapsed time

License is Public Domain

Contains an estimated 60 Million entries

### Sample data: 
| FL_DATE   | OP_CARRIER | OP_CARRIER_FL_NUM | ORIGIN | DEST | CRS_DEP_TIME | DEP_TIME | DEP_DELAY | TAXI_OUT | WHEELS_OFF | CRS_ELAPSED_TIME | ACTUAL_ELAPSED_TIME | AIR_TIME | DISTANCE | CARRIER_DELAY | WEATHER_DELAY | NAS_DELAY | SECURITY_DELAY | LATE_AIRCRAFT_DELAY |
| --------- | ---------- | ----------------- | ------ | ---- | ------------ | -------- | --------- | -------- | ---------- | ---------------- | ------------------- | -------- | -------- | ------------- | ------------- | --------- | -------------- | ------------------- |
| 2018-01-01| UA | 2429 | EWR | DEN | 1517 | 1512.0 | -5.0 | 15.0 | 1527.0 | 268.0 | 250.0 | 225.0 | 1605.0 | NaN | NaN | NaN | NaN | NaN |
| 2018-01-01| UA | 2427 | LAS | SFO | 1115 | 1107.0 | -8.0 | 11.0 | 1118.0 | 99.0  | 83.0  | 65.0  | 414.0  | NaN | NaN | NaN | NaN | NaN |
| 2018-01-01| UA | 2426 | SNA | DEN | 1335 | 1330.0 | -5.0 | 15.0 | 1345.0 | 134.0 | 126.0 | 106.0 | 846.0  | NaN | NaN | NaN | NaN | NaN |
| 2018-01-01| UA | 2425 | RSW | ORD | 1546 | 1552.0 | 6.0  | 19.0 | 1611.0 | 190.0 | 182.0 | 157.0 | 1120.0 | NaN | NaN | NaN | NaN | NaN |

⬆️ [To summary table](#summary-table)

## OTS Data

Detailed and complete data provided by the United States Office of Transportation Safety. It is made available [here](https://www.transtats.bts.gov) aviation -> Airline On-Time Performance Data -> Marketing Carrier On-Time Performance (Beginning January 2018).

Recommended to download the whole zipped file (otherwise website issues). Button is just under the description on the right.
Data is available per month and updated regularly.

It contains several databases collected mostly through surveys and studies and, among others, makes the following available:
- Airline On-Time Performance Data: 
- Freight and passenger statistics
- Ticket surveys
- Intermodal passenger connectivity

See website for full list or see [here](https://www.kaggle.com/datasets/bordanova/2023-us-civil-flights-delay-meteo-and-aircraft) for already pre-processed examples
### Airline On-Time Performance Data:
- 120 columns
- Flight Schedule with details for delays and diversion

| DOT_ID_Operating_Airline | Tail_Number | OriginAirportID | OriginCityName | DestAirportID | DestCityName | DepTime | DepDelay | WheelsOff | ArrDelay |
| ------------------------ | ----------- | --------------- | -------------- | ------------- | ------------ | ------- | -------- | --------- | -------- |
| 20500 | N535GJ | 13296 | Manchester, NH | 11618 | Newark, NJ   | 1849.0  | 71.0 | 1913.0 | 45.0 |
| 20500 | N535GJ | 12264 | Washington, DC | 11618 | Newark, NJ   | 814.0   | -1.0 | 912.0  | 26.0 |
| 20500 | N535GJ | 11618 | Newark, NJ     | 13296 | Manchester, NH| 1654.0 | 74.0 | 1735.0 | 86.0 |
| 20500 | N547GJ | 15016 | St. Louis, MO  | 13930 | Chicago, IL  | 630.0   | 0.0  | 717.0  | 25.0 |
| 20500 | N504GJ | 15016 | St. Louis, MO  | 12264 | Washington, DC| 1333.0 | 33.0 | 1345.0 | 18.0 |


See more details in the [OTS vizualisations](/OTS/ots.ipynb) or [OTS README file](/OTS/README.md)

⬆️ [To summary table](#summary-table)

## PlaneFinder

🟠 Had been contacted for samples

[Website](https://planefinder.net/data)

A commercial plaform providing curated datasets on: 
- Flights: ADSB like, flight tracking and flight status
- Airports: Schedules, departires, arrivals and Local weather
- Airlines: Fleet information, aviation photography...

⬆️ [To summary table](#summary-table)


## AvDelphi

A mostly aviation metadata oriented online database with various information on
Contains valuable information for fine grain manual research but isn't suited to big data.

- Supliers
- ACARS Records
- Aircraft types
- Airframes
- Airlines and Operators
- Airports information (API Compatible)
- ATC & FIR zones
- Aviation Comms
- AWD's
- Flights Tables & metadata (API Compatible)
- Images
- NOTAMS (API Compatible)
- Powerplants
- Radar Sites
- Waypoints
- Weather information


## ACI World

[The Annual World Airport Traffic Dataset, 2023 Edition](https://store.aci.aero/product/annual-world-airport-traffic-dataset-2023/), is the industry’s most comprehensive airport statistics dataset featuring airport traffic for over 2,600 airports across more than 180 countries and territories.
It provides a view of air transport demand across the world’s airports by three thematic areas: passengers (international and domestic), air cargo (freight and mail) and aircraft movements (air transport movements and general aviation) for the year 2022.

A total of ~2600 aiports are represented, and numbers account for both year 2022 and 2021.

⬆️ [To summary table](#summary-table)

### Table Headers Description

Header | Description
--- | ---
Region | The geographical region where the airport is located
Country | The country where the airport is located
Country Code | The ISO country code of the country where the airport is located
City | The city where the airport is located
Airport Name | The official name of the airport
IATA Code | The International Air Transport Association code for the airport
Movements Combi 2022 | The combined movements at the airport in 2022
Movements Cargo/mail 2022 | The number of cargo/mail movements at the airport in 2022
Movements Air transport 2022 | The number of air transport movements at the airport in 2022
Movements General aviation and Military 2022 | The number of general aviation and military movements at the airport in 2022
Movements 2022 | The total number of movements at the airport in 2022
Passenger International 2022 | The number of international passengers at the airport in 2022
Passenger Domestic 2022 | The number of domestic passengers at the airport in 2022
Passenger Terminal 2022 | The number of passengers at the airport terminal in 2022
Passenger Direct transit 2022 | The number of direct transit passengers at the airport in 2022
Passengers 2022 | The total number of passengers at the airport in 2022
Freight International 2022 | The amount of international freight handled at the airport in 2022
Freight Domestic 2022 | The amount of domestic freight handled at the airport in 2022
Freight 2022 | The total amount of freight handled at the airport in 2022
Mail 2022 | The amount of mail handled at the airport in 2022
Total cargo 2022 | The total amount of cargo handled at the airport in 2022
... 2021 data ... |  Similar information as above for year 2021 

⬆️ [To summary table](#summary-table)

## OpenSky

Freely accessible ADS-B data [here](https://opensky-network.org), and one of the largest of its kind. Quality compares to the commercial data available on other platforms.

OpenSky also provides a Live API for aircraft tracking.

In the future, ATC voice recordings will be made available.

Is only availble for research by universities or other 'not-for-profit' institutions.

>Contains: 
>- Historical raw ADS-B flight record data
>- API access to live tracking of planes
>- Scientific [set of datasets](https://opensky-network.org/data/datasets#d2) (Vector states, Raw data, LocaRDS, Covid19, Metadata, Emergencies, Climbings, ADS-C ...)


### Note:
Currently, the registration/user database is migrated and being worked on. Registration and emails may not work properly for a few days and new accounts could get lost. If you urgently need access, please contact us.


### Description: 
The OpenSky Network is a non-profit community-based receiver network which has been continuously collecting air traffic surveillance data since 2013. Unlike other networks, OpenSky keeps the complete unfiltered raw data and makes it accessible to academic and institutional researchers. With over 30 trillion ADS-B, Mode S, TCAS and FLARM messages collected from more than 6000 sensors around the world, the OpenSky Network exhibits the largest air traffic surveillance dataset of its kind. The mission of our non-profit association is to support open global air traffic research by universities and other not-for-profit institutions.

⬆️ [To summary table](#summary-table)

## Data World

A [collection of small datasets](https://data.world/datasets/aviation) of varying nature such as.
- Airport information for every country (size, geo, names)
- Some flight schedules eg ([this](https://data.world/city-of-phoenix/cc84a16c-344d-4196-8441-130a5f6be607) or [this](https://data.world/hoytick/2017-jan-ontimeflightdata-usa))
- Flight Information Regions (FIR)
Data is very different in format, size. Requires a lot of research and pre-processing

⬆️ [To summary table](#summary-table)

## KiltHub - TrajAir

Public Domain dataset of general aviation flights, acquired between `12/09/2020`and `27th/04/2021`. It is available for download [here](https://kilthub.cmu.edu/articles/dataset/TrajAir_A_General_Aviation_Trajectory_Dataset/14866251)

Total rows: `2 731 256`

### Data sample:

| ID  | Time    | Date  | Altitude | Speed | Heading | Lat  | Lon  | Age | Range | Bearing | Tail | Metar |
| --------- | ----------- | --------- | -------- | -------- | -------- | --------- | --------- | -------- | ------------------- | ------------------- | --------- | --------- |
| 10837576  | 06:56:47.554| 09/18/2020| 2400     | 132      | 152      | 40.753006 | -80.17804 | 1.707318 | 19.46265851422735   | -98.08739957853011  | N445ME    | K ... |
| 10837576  | 06:56:49.451| 09/18/2020| 2400     | 132      | 152      | 40.753006 | -80.17804 | 0.708735 | 19.46265851422735   | -98.08739957853011  | N445ME    | K ... |
| 10837576  | 06:56:52.423| 09/18/2020| 2400     | 132      | 152      | 40.749687 | -80.17567 | 0.591813 | 19.32137670596792   | -99.25419554890154  | N445ME    | K ... |
| 10837576  | 06:56:53.542| 09/18/2020| 2400     | 132      | 152      | 40.74774  | -80.17432 | 0.685392 | 19.24561039228262   | -99.94469440923552  | N445ME    | K ... |

⬆️ [To summary table](#summary-table)