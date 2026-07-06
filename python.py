import ee
import requests
import os
import csv

ee.Initialize(project='ee-radhamaheshdhuri')

output_folder = 'tif'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

csv_path = os.path.join(output_folder, 'metadata.csv')

# Study area (square)
point = ee.Geometry.Point([15.31, -24.76])
study_area = point.buffer(3000).bounds()

# Define date ranges
years = range(2017, 2027)  # 2017 to 2026 inclusive
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # All months

bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']


# Function to get best image for a given year-month
def get_best_image(year, month):
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-28'

    collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                  .filterBounds(study_area)
                  .filterDate(start_date, end_date)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))

    best = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
    return best


# CSV field names
fieldnames = ['filename', 'year', 'month', 'date', 'cloudy_pixel_percentage',
              'mean_solar_azimuth_angle', 'mean_solar_zenith_angle']

# Open CSV once, write incrementally so partial progress isn't lost on a crash
csv_file = open(csv_path, 'w', newline='')
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()
csv_file.flush()

downloaded = 0
failed = 0

for year in years:
    for month in months:
        print(f"\nProcessing: {year}-{month:02d}")

        try:
            image = get_best_image(year, month)

            if image is None:
                print(f"  ✗ No image found for {year}-{month:02d}")
                failed += 1
                continue

            date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
            clouds = image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()

            print(f"  Image date: {date} (clouds: {clouds:.1f}%)")

            # Pull extra metadata properties (may be missing on some images)
            info = image.toDictionary(
                ['MEAN_SOLAR_AZIMUTH_ANGLE', 'MEAN_SOLAR_ZENITH_ANGLE']
            ).getInfo()
            sun_azimuth = info.get('MEAN_SOLAR_AZIMUTH_ANGLE')
            sun_zenith = info.get('MEAN_SOLAR_ZENITH_ANGLE')

            # Download
            url = image.getDownloadURL({
                'region': study_area,
                'bands': bands,
                'scale': 10,
                'format': 'GEOTIFF'
            })

            response = requests.get(url)
            filename = os.path.join(output_folder, f'sossusvlei_{year}_{month:02d}_{date}.tif')

            with open(filename, 'wb') as f:
                f.write(response.content)

            print(f"  ✓ Saved: {filename} ({len(response.content)/1e6:.1f} MB)")
            downloaded += 1

            # Build CSV row
            row = {
                'filename': os.path.basename(filename),
                'year': year,
                'month': month,
                'date': date,
                'cloudy_pixel_percentage': clouds,
                'mean_solar_azimuth_angle': sun_azimuth,
                'mean_solar_zenith_angle': sun_zenith,
            }

            writer.writerow(row)
            csv_file.flush()

        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed += 1

csv_file.close()

print("\n" + "="*60)
print(f"COMPLETE!")
print(f"  Downloaded: {downloaded} images")
print(f"  Failed: {failed} images")
print(f"  Metadata CSV: {csv_path}")
print("="*60)