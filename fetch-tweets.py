from credentials import CONSUMER_KEY, CONSUMER_SECRET
import tweepy, json, argparse
from time import sleep

def fetch_tweets(rows):
  tweet_ids = [row['original_id'] for row in rows]
  if args.gifs:
    tweet_ids.extend([row['reply_id'] for row in rows])
  return {tweet.id_str:tweet for tweet in fetch_ids(tweet_ids) if tweet.id_str != ''}

def fetch_ids(ids):
  id_lists = [ids[x:x+100] for x in range(0, len(ids), 100)]
  tweets = []
  for idx, id_list in enumerate(id_lists):
    print('{}.'.format(len(id_lists) - idx), flush=True, end='')
    tweets.extend([tweet for tweet in api.statuses_lookup(id_list, tweet_mode='extended')])
  print()
  return tweets

def deanon(de_anon_file, rows, tweets):
  found = 0
  with open(de_anon_file, 'w') as f:
    for row in rows:
      if row['original_id'] in tweets:
        found += 1
        row['text'] = tweets[row['original_id']].full_text
        if args.gifs:
          if row['reply_id'] in tweets:
            row['reply'] = tweets[row['reply_id']].extended_entities['media'][0]['video_info']['variants'][0]['url']
          else:
            row['reply'] = None
        f.write((json.dumps(row, ensure_ascii=False) + "\n"))
  print(f'Found {found} tweets out of {len(rows)}.')

def convert(anon_file, de_anon_file):
    print('Fetching texts for {}'.format(anon_file))
    rows = [json.loads(row) for row in open(anon_file, 'r').readlines()]
    if args.limit != -1:
      rows = rows[0:args.limit]
    tweets = fetch_tweets(rows)
    deanon(de_anon_file, rows, tweets)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Fetch tweets.')
  parser.add_argument('--gifs', action='store_true', help='Add links to reaction GIFs')
  parser.add_argument('--limit', default=-1, type=int, help='Number of tweets to fetch')
  args = parser.parse_args()

  try:
    CONSUMER_KEY
    CONSUMER_SECRET
  except:
    print('Edit credentials.py to add your Twitter API credentials in the first two lines (CONSUMER_KEY and CONSUMER_SECRET)')
    print('See here for more information on getting API credentials: https://developer.twitter.com/en/apps')
    exit(1)
  auth = tweepy.AppAuthHandler(CONSUMER_KEY , CONSUMER_SECRET)
  api = tweepy.API(auth, wait_on_rate_limit=True,
                        wait_on_rate_limit_notify=True,
                        retry_count=10, retry_delay=60,
                        retry_errors=[400] + list(range(402,599)))
  print('This can take some time, so make yourself a cup of Taiwanese oolong tea and let the magic happen!')
  sleep(3)
  convert('ReactionGIF.ids.json', 'ReactionGIF.json')
  print('That\'s it! Hope you enjoyed the ride :)')
