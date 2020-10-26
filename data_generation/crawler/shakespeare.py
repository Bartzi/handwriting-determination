import requests
from bs4 import BeautifulSoup
import urllib
import tqdm

BASE_PAGE_URL = 'https://luna.folger.edu/luna/servlet/view/search?q=Call_Number%3DL.c.*&os={}'
SAVE_PATH = 'downloaded_images'


def download_images():
    for results_number in tqdm.tqdm(range(0, 12485, 250)):
        page_url = BASE_PAGE_URL.format(results_number)

        page = requests.get(page_url)

        # Create a BeautifulSoup object
        soup = BeautifulSoup(page.text, 'html.parser')

        images = soup.findAll('img', style='max-height:192px; overflow-x:hidden;')

        for image_node in images:
            small_image_url = image_node.attrs['src']
            large_image_url = small_image_url.replace("Size1", "Size4")

            file_name = large_image_url.rsplit('/', 1)[-1]

            urllib.request.urlretrieve(large_image_url, "{}/{}".format(SAVE_PATH, file_name))


if __name__ == '__main__':
    download_images()
