"""Just some example code for getting images via the twitter API"""

import twitter
import json


def to_date(created_at):
    import dateutil.parser
    datetime = dateutil.parser.parse(created_at)
    return "{:%Y-%m-%d}".format(datetime)

# for docs how to generate your own authentication file, see:
# https://python-twitter.readthedocs.io/en/latest/getting_started.html
with open('twitter_auth.json', 'r') as auth_file:
    auth = json.load(auth_file)

api = twitter.Api(**auth)

s = api.GetSearch(raw_query='l=&q=from%3Ahourlyfox since%3A2017-01-01&count=100')

image_url = json.loads(s[0].AsJsonString())['media'][0]['media_url']

oldest_date = json.loads(s[-1].AsJsonString())['created_at']

s = api.GetSearch(raw_query='l=&q=from%3Ahourlyfox until%3A' + to_date(oldest_date) + '&count=100')


