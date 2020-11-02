import argparse
import json
from datetime import datetime

import dateutil.parser as dparser
import pandas as pd
from pathlib import Path

from logger import logger

THRESHOLD_SCORE = 3


def process_fakehealth(path: str):
    dir = Path(path)
    reviews = pd.read_json(dir / 'reviews/HealthStory.json')
    fake_stories = reviews[reviews["rating"] < THRESHOLD_SCORE]
    true_stories = reviews[reviews["rating"] >= THRESHOLD_SCORE]

    logger.info("Stats of Health Story")
    logger.info(f"Num of fake stories {len(fake_stories)}")
    logger.info(f"Num of true stories {len(true_stories)}")
    logger.info(f"Num of unique sources in fake stories {len(fake_stories.news_source.unique())}")
    logger.info(f"Num of unique sources in true stories {len(true_stories.news_source.unique())}")
    logger.info(
        f"Num of overlapped sources {len(set(fake_stories.news_source.unique()) - set(true_stories.news_source.unique()))}")

    filtered_fake_stories = pd.DataFrame(extract_content(dir, "HealthStory", fake_stories.news_id.values, "fake"))
    filtered_true_stories = pd.DataFrame(extract_content(dir, "HealthStory", true_stories.news_id.values, "true"))

    releases = pd.read_json(dir / 'reviews/HealthRelease.json')
    fake_releases = releases[releases["rating"] < THRESHOLD_SCORE]
    true_releases = releases[releases["rating"] >= THRESHOLD_SCORE]

    logger.info("Stats of Health Releases")
    logger.info(f"Num of fake releases {len(fake_stories)}")
    logger.info(f"Num of true releases {len(true_stories)}")
    logger.info(f"Num of unique sources in fake releases {len(fake_releases.news_source.unique())}")
    logger.info(f"Num of unique sources in true releases {len(true_releases.news_source.unique())}")
    logger.info(
        f"Num of overlapped sources {len(set(fake_stories.news_source.unique()) - set(true_stories.news_source.unique()))}")

    filtered_fake_releases = pd.DataFrame(extract_content(dir, "HealthRelease", fake_releases.news_id.values, "fake"))
    filtered_true_releases = pd.DataFrame(extract_content(dir, "HealthRelease", true_releases.news_id.values, "true"))
    merged_data = pd.concat(
        [filtered_fake_releases, filtered_true_stories, filtered_fake_stories, filtered_true_releases])

    Path('Data/Processed').mkdir(parents=True, exist_ok=True)
    processed_dir = Path('Data/Processed')
    merged_data.to_csv(processed_dir / 'FakeHealth.tsv', sep='\t')
    logger.info(f"Merged data is saved to {processed_dir / 'FakeHealth.tsv'}")


def extract_content(dir, news_source, news_ids, label):
    processed_data = []
    for news_id in news_ids:
        data = {}
        fname = f"content/{news_source}/{news_id}.json"
        if not (dir / fname).exists():
            logger.error(f"{news_id} does not exist")
            continue
        with open(dir / fname) as f:
            news_item = json.load(f)
            data["content"] = news_item["text"]
            if news_item["publish_date"]:
                data["publish_date"] = datetime.fromtimestamp(news_item["publish_date"])
            elif "lastPublishedDate" in news_item["meta_data"]:
                data["publish_date"] = news_item["meta_data"]["lastPublishedDate"]
            elif "dcterms.modified" in news_item["meta_data"]:
                data["publish_date"] = news_item["meta_data"]["dcterms.modified"]
            elif "article.updated" in news_item["meta_data"]:
                data["publish_date"] = news_item["meta_data"]["article.updated"]
            elif "date" in news_item["meta_data"]:
                data["publish_date"] = news_item["meta_data"]["date"]
            else:
                try:
                    data["publish_date"] = dparser.parse(news_item["text"][:100], fuzzy=True)
                except:
                    logger.error(f"{news_id} does not have a publish_date")
                    continue
            data["url"] = news_item["url"]
            data["news_id"] = news_id
            data["label"] = label
            processed_data.append(data)
    logger.info(f"Num of processed {label} news with date {len(processed_data)}")
    return processed_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fakehealth', type=str, help="Directory path of FakeHealth")
    args = parser.parse_args()

    if args.fakehealth:
        process_fakehealth(args.fakehealth)
