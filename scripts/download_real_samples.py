import requests
import os

os.makedirs('data/real_samples', exist_ok=True)
stac_api = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'

def download_preview(url, filename):
    print(f'Downloading {filename}...')
    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)

# Bounding box for a region in India (New Delhi area)
bbox = [77.0, 28.0, 77.5, 28.5]

print('Fetching cloud-free reference...')
search_clear = {
    'collections': ['sentinel-2-l2a'],
    'bbox': bbox,
    'datetime': '2023-01-01/2023-12-31',
    'query': {'eo:cloud_cover': {'lt': 5}},
    'limit': 1
}
resp_clear = requests.post(stac_api, json=search_clear).json()
if 'features' in resp_clear and len(resp_clear['features']) > 0:
    clear_item = resp_clear['features'][0]
    clear_url = clear_item['assets']['rendered_preview']['href']
    download_preview(clear_url, 'data/real_samples/clear_reference.png')
else:
    print('No clear images found.')

print('Fetching cloudy samples...')
search_cloudy = {
    'collections': ['sentinel-2-l2a'],
    'bbox': bbox,
    'datetime': '2023-01-01/2023-12-31',
    'query': {'eo:cloud_cover': {'gt': 30, 'lt': 80}},
    'limit': 5
}
resp_cloudy = requests.post(stac_api, json=search_cloudy).json()
if 'features' in resp_cloudy:
    for i, item in enumerate(resp_cloudy['features']):
        cloudy_url = item['assets']['rendered_preview']['href']
        download_preview(cloudy_url, f'data/real_samples/cloudy_{i}.png')
else:
    print('No cloudy images found.')

print('Done.')
