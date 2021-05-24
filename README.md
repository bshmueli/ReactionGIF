# ReactionGIF

ReactionGIF is a unique, first-of-its-kind dataset of 30K sarcastic tweets and their GIF reactions. 

To find out more about ReactionGIF, 
check out our ACL 2021 paper:

* Shmueli, Ray and Ku, [Happy Dance, Slow Clap: Using Reaction GIFs to Predict Induced Affect on Twitter](https://arxiv.org/abs/2105.09967)

Use this repository to download ReactionGIF. The repository includes the following data files:

  * `ReactionGIF.ids.json` with original tweet IDs in [jsonlines](https://jsonlines.org) format.

The fields included for each tweet are:
* ``idx`` record number (note: record numbers are not sequential)
* `original_id` the tweet ID of the original tweet which contains the eliciting text
* ``reply_id`` the tweet ID of the reply tweet which contains the reaction GIF
* ``label`` the reaction category

To comply with Twitter's privacy policy, the dataset  include only the tweet IDs. To fetch the original tweets' texts, follow these steps:

  1. Install the latest version of Tweepy:
  
    pip3 install tweepy
  2. Rename our `credentials-example.py` to `credentials.py`

    mv credentials-example.py credentials.py
  4. Add your [Twitter API credentials](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api) by editing `credentials.py`

    vim credentials.py
  6. Run the script:
  
    python3 fetch-tweets.py

The script will fetch the tweet texts and add a new ``text`` field, saving the new dataset to:

  * `ReactionGIF.json`

## Citation

If you use our dataset, kindly cite the paper using the following BibTex entry:

```
@misc{shmueli2021happy,
      title={Happy Dance, Slow Clap: Using Reaction {GIFs} to Predict Induced Affect on {Twitter}}, 
      author={Boaz Shmueli and Soumya Ray and Lun-Wei Ku},
      year={2021},
      eprint={2105.09967},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

