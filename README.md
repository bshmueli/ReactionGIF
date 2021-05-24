# ReactionGIF

ReactionGIF is a unique dataset of 30K sarcastic tweets and their GIF reactions. 

To find out more about ReactionGIF, 
check out our paper ["Happy Dance, Slow Clap: Using Reaction GIFs to Predict Induced Affect on Twitter"](https://arxiv.org/abs/2105.09967) (soon to appear in ACL 2021).

Use this repository to download ReactionGIF. The repository includes the following data files:

  * `ReactionGIF.ids.json` with original tweet IDs in [jsonlines](https://jsonlines.org) format.

The fields included for each tweet are:
* ``idx``: record number (note: record numbers are not sequential)
* `original_id`: the tweet ID of the original tweet which contains the eliciting text
* ``reply_id``: the tweet ID of the reply tweet which contains the reaction GIF
* ``label``: the reaction category

To comply with Twitter's privacy policy, the dataset  include only the tweet IDs. To fetch the original tweets' texts, follow these steps:

  * Install the latest version of Tweepy:
  
    `pip3 install tweepy`
  * Rename our `credentials-example.py` to `credentials.py`
  * Add your Twitter API credentials by editing `credentials.py`
  * Run the script:
  
    `python3 fetch-tweets.py`

The script will fetch the texts and create a new file:

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

