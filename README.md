# ReactionGIF
The ReactionGIF dataset of tweets and their GIF reactions

ReactionGIF is a unique dataset of around 30K sarcastic tweets and their GIF reactions. 

To find out more about ReactionGIF, check out the ["Happy dance, slow clap"](https://arxiv.org/abs/???) paper.

Use this repository to download ReactionGIF. The repository includes the following data files:

  * `ReactionGIF-ids.json` with original tweet IDs in [jsonlines](https://jsonlines.org) format.
  
Additional fields for each original tweet include the ID of the reply tweet containing the reaction GIF, and the reaction category of the GIF.
More information is available in the "Happy dance, slow clap" paper.

To comply with Twitter's privacy policy, the dataset  include only the tweet IDs. To fetch the original tweet texts, follow these steps:

  * Install the latest version of Tweepy:
  
    `pip3 install tweepy`
  * Rename our `credentials-example.py` to `credentials.py`
  * Add your Twitter API credentials by editing `credentials.py`
  * Run the script:
  
    `python3 fetch-tweets.py`

The script will fetch the texts and create a new file:

  * `ReactionGIF.json`

## Citation

Kindly cite the paper using the following BibTex entry:

```
@inproceedings{
}
```

