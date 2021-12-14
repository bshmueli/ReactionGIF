# ReactionGIF

ReactionGIF is a unique, first-of-its-kind dataset of 30K sarcastic tweets and their GIF reactions. 

To find out more about ReactionGIF, 
check out our ACL 2021 paper:

* Shmueli, Ray and Ku, [Happy Dance, Slow Clap: Using Reaction GIFs to Predict Induced Affect on Twitter](https://aclanthology.org/2021.acl-short.50/)

Use this repository to download ReactionGIF. The repository includes the following data file with the tweet information:

  * `ReactionGIF.ids.json` with original tweet IDs in [jsonlines](https://jsonlines.org) format.

Each record in the file includes the following fields:
* ``idx`` record number (note: record numbers are not sequential)
* `original_id` the tweet ID of the original tweet which contains the eliciting text
* ``reply_id`` the tweet ID of the reply tweet which contains the reaction GIF
* ``label`` the reaction category

To comply with Twitter's [ToS](https://twitter.com/tos) and [Developer Agreement and Policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy), the dataset  includes only the tweet IDs. To fetch the original tweets' texts, you can write your own script, or you can use our own, easy-to-use script. To use our own script, follow these steps:

  1. Clone the repository or download the files `ReactionGIF.ids.json`, `credentials-example.py`, and `fetch-tweets.py`
  2. Install the latest version of [Tweepy](https://www.tweepy.org):
  
    pip3 install tweepy
  3. Rename our `credentials-example.py` to `credentials.py`

    mv credentials-example.py credentials.py
  4. Add your [Twitter API credentials](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api) by editing `credentials.py`:

    vim credentials.py
  5. Run the script:
  
    python3 fetch-tweets.py [--gifs]

The script will fetch the tweet texts and add a new ``text`` field. If you turn on the `gifs` flag, the script will also add a ``reply`` field which will include the link to the GIF. The new dataset will be saved to:

  * `ReactionGIF.json`

## Citation

If you use our dataset, kindly cite the paper using the following BibTex entry:

```
@inproceedings{shmueli-etal-2021-happy,
    title = "Happy Dance, Slow Clap: {Using} Reaction {GIFs} to Predict Induced Affect on {Twitter}",
    author = "Shmueli, Boaz  and
      Ray, Soumya  and
      Ku, Lun-Wei",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-short.50",
    doi = "10.18653/v1/2021.acl-short.50",
    pages = "395--401",
    abstract = "Datasets with induced emotion labels are scarce but of utmost importance for many NLP tasks. We present a new, automated method for collecting texts along with their induced reaction labels. The method exploits the online use of reaction GIFs, which capture complex affective states. We show how to augment the data with induced emotion and induced sentiment labels. We use our method to create and publish ReactionGIF, a first-of-its-kind affective dataset of 30K tweets. We provide baselines for three new tasks, including induced sentiment prediction and multilabel classification of induced emotions. Our method and dataset open new research opportunities in emotion detection and affective computing.",
}
```

