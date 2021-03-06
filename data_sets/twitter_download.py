"""
Downloading images via the Twitter API

for docs how to generate your own authentication file, see:
https://python-twitter.readthedocs.io/en/latest/getting_started.html

"""
from urllib.error import HTTPError

import twitter
import json

from os.path import join, isfile
from os import makedirs

from PIL import Image
from urllib.request import urlretrieve

MAX_IMAGES = 540
TWITTER_ACCOUNTS = [
    'hourlyfox', 'Bodegacats_', 'BirdPerHour', 'ravenmaster1', 'HourlyPinguins', 'HourlyPanda'
]


class TwitterDownloader:

    def __init__(self, api, account, max_images=MAX_IMAGES):
        self.api = self._resolve_twitter_api(api)
        self.account = account
        self.max_images = max_images
        self.image_urls = set()
        self.download_root = join('.', 'data', 'twitter')

    def get_images(self, download=False):
        search_results = self.api.GetUserTimeline(
            screen_name=self.account, count=200, include_rts=False, exclude_replies=True
        )
        while len(self.image_urls) < self.max_images and search_results:
            max_id = json.loads(search_results[-1].AsJsonString())['id']
            for post in search_results:
                try:
                    self.add_images_in_post(post, download)
                except (IndexError, KeyError):
                    continue

            search_results = self.api.GetUserTimeline(
                screen_name=self.account, count=200, include_rts=False, exclude_replies=True,
                max_id=max_id-1
            )

        return list(self.image_urls)[:self.max_images]

    def add_images_in_post(self, post, download=False):
        for media in json.loads(post.AsJsonString())['media']:
            image_url = media['media_url']
            if image_url[-4:] in ('.jpg', '.png'):
                try:
                    if download:
                        self.download_image(image_url)
                    self.image_urls.add(image_url)
                except HTTPError:
                    pass

    def download_image(self, url):
        from urllib.error import URLError
        from os import remove
        makedirs(join(self.download_root, self.account), exist_ok=True)
        filename = join(self.download_root, self.account, url.split('/')[-1])
        try:
            if not isfile(filename):
                urlretrieve(url, filename)
            with Image.open(filename) as image:
                # image.show()
                print(filename, image.size)
        except (URLError, ConnectionResetError, OSError):
            try:
                remove(filename)
            except FileNotFoundError:
                pass

    @staticmethod
    def _resolve_twitter_api(api):
        if isfile(api):
            with open(api, 'r') as auth_file:
                auth = json.load(auth_file)
            api = twitter.api.Api(**auth, sleep_on_rate_limit=True)
        if not isinstance(api, twitter.Api):
            raise ValueError(
                'api parameter must either be a file with authorization parameters or an instantiated twitter API'
            )
        if not api.VerifyCredentials():
            raise ValueError('Twitter API credentials not valid')
        return api


image_urls = {
    account: TwitterDownloader('twitter_auth.json', account).get_images(True)
    for account in TWITTER_ACCOUNTS
}
