import requests
import logging

class ImageProviderClient:
    BASE_URL = 'http://89.169.157.72:8080/images/'
    TIMEOUT = 5 

    @staticmethod
    def get_image(image_id: int) -> bytes:
        url = f'{ImageProviderClient.BASE_URL}{image_id}'
        try:
            response = requests.get(url, timeout=ImageProviderClient.TIMEOUT)
            response.raise_for_status()
            if 'image' not in response.headers.get('Content-Type', ''):
                raise ValueError('Not an image file')
            return response.content
        except (requests.RequestException, ValueError) as e:
            logging.error(f'Failed to fetch image {image_id}: {e}')
            raise
