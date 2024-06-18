import cdsapi
import cfgrib
import xarray as xr

def download_data(year:int, month, day, hour, pressure_level:int, output_file:str = 'download.grib', **kwargs):
    year = '{:04d}'.format(year)
    month = '{:02d}'.format(month)
    day = '{:02d}'.format(day)
    time = '{:02d}:00'.format(hour)
    pressure_level = str(pressure_level)


    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'grib',
            'variable': [
                'relative_humidity', 'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
                'specific_humidity', 'specific_snow_water_content', 'temperature',
                'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
            ],
            # 'area':['north', 'west', 'south', 'east'] # North, West, South, East
            'month': [month],
            'year': [year],
            'day': [day],
            'time': [time],
            'pressure_level': [pressure_level],
        },
        output_file)


def calculate_level_wind_speed(u,v):
    """ Calculate the wind speed at a given level
    Args:
        u: u component of wind
        v: v component of wind
    """
    return (u**2 + v**2)**0.5

def retrieve_closest_point(timestamp, longitude, latitude, altitude_ft):
    """ Retrieve the closest point in the ERA5 dataset to the given coordinates
    Args:
        timestamp: datetime object
        longitude: float
        latitude: float
        altitude_ft: int
    """
    info = {
        'year': timestamp.year,
        'month': timestamp.month,
        'day': timestamp.day,
        'hour': timestamp.hour,
        'longitude': round(longitude*4)/4,
        'latitude': round(latitude*4)/4,
        'pressure_level': round((1013.25 * (1 - 2.25577e-5 * altitude_ft * 0.3048)**5.25588)/25)*25
    }
    return info


from datetime import datetime
time = datetime.now()
print(type(time))
timestamp = "2023-04-01 12:00:00"
time = datetime.timestamp(datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S'))
time = datetime.fromtimestamp(time)

print(retrieve_closest_point(time, 12.5678490, 67.8765, 30480))

download_data(**retrieve_closest_point(time, 12.5678490, 67.8765, 30480))


# c = cdsapi.Client()

# c.retrieve(
#     'reanalysis-era5-pressure-levels',
#     {
#         'product_type': 'reanalysis',
#         'format': 'grib',
#         'variable': [
#             'relative_humidity', 'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
#             'specific_humidity', 'specific_snow_water_content', 'temperature',
#             'u_component_of_wind', 'v_component_of_wind',
#         ],
#         'month': '04',
#         'year': '2023',
#         'day': [
#             '01', '02',
#         ],
#         'time': [
#             '00:00', '01:00', '02:00',
#             '03:00', '04:00', '05:00',
#             '06:00', '07:00', '08:00',
#             '09:00', '10:00', '11:00',
#             '12:00', '13:00', '14:00',
#             '15:00', '16:00', '17:00',
#             '18:00', '19:00', '20:00',
#             '21:00', '22:00', '23:00',
#         ],
#         'pressure_level': [
#             '250', '850', '975',
#         ],
#     },
#     'download.grib')

# import pygrib
# import numpy as np
# ds = xr.open_dataset('download.grib', engine='cfgrib')
# latitudes = np.unique(ds.coords['longitude'].values)
# print(latitudes)
# df:pd.DataFrame = ds.to_dataframe()

# df.info()
