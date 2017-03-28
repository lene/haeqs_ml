"""Just some example code for getting images via the twitter API"""

import twitter
import json

from os.path import join
from os import makedirs

from PIL import Image
from urllib.request import urlretrieve

MAX_IMAGES = 540
TWITTER_ACCOUNTS = [
    'hourlyfox', 'Bodegacats_', 'BirdPerHour', 'ravenmaster1', 'HourlyPinguins', 'HourlyPanda'
]


def to_date(created_at):
    import dateutil.parser
    return "{:%Y-%m-%d}".format(dateutil.parser.parse(created_at))


def crawl_twitter_account(twitter_api, twitter_account):
    image_urls = set()
    search_results = twitter_api.GetUserTimeline(
        screen_name=twitter_account, count=200, include_rts=False, exclude_replies=True
    )
    while len(image_urls) < MAX_IMAGES and search_results:
        oldest_date = json.loads(search_results[-1].AsJsonString())['created_at']
        max_id = json.loads(search_results[-1].AsJsonString())['id']
        for post in search_results:
            try:
                for media in json.loads(post.AsJsonString())['media']:
                    image_url = media['media_url']
                    if image_url[-4:] in ('.jpg', '.png'):
                        image_urls.add(image_url)
            except (IndexError, KeyError):
                continue

        search_results = twitter_api.GetUserTimeline(
            screen_name=twitter_account, count=200, include_rts=False, exclude_replies=True,
            max_id=max_id-1
        )
        print(max_id, oldest_date)

    return image_urls

# for docs how to generate your own authentication file, see:
# https://python-twitter.readthedocs.io/en/latest/getting_started.html
with open('twitter_auth.json', 'r') as auth_file:
    auth = json.load(auth_file)

api = twitter.api.Api(**auth, sleep_on_rate_limit=True)

image_urls = {account: crawl_twitter_account(api, account) for account in TWITTER_ACCOUNTS}

from pprint import pprint
pprint({k: len(v) for k, v in image_urls.items()})

account = 'ravenmaster1'
for url in image_urls[account]:
    makedirs(join('data', 'twitter', account), exist_ok=True)
    filename = join('data', 'twitter', account, url.split('/')[-1])
    urlretrieve(url, filename)
    image = Image.open(filename)
    # image.show()
    print(filename, image.size)

