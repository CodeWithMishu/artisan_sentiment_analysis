"""Connector: run snscrape, map results to district(s), and POST to /api/ingest

Usage: python scripts/connector_snscrape_ingest.py --query "#handmade OR #kangra" --district kangra --batch 20

This script requires snscrape to be installed in the environment.
"""
import argparse
import subprocess
import json
import requests
import sys
import tempfile
import os
from datetime import datetime


INGEST_URL = os.environ.get('INGEST_URL', 'http://127.0.0.1:5000/api/ingest')


def run_snscrape(query, max_results=200):
    # Use snscrape to produce JSONL
    try:
        proc = subprocess.run(['snscrape','--jsonl','twitter-search', query], capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            print('snscrape failed:', proc.stderr)
            return []
        lines = proc.stdout.splitlines()
        results = [json.loads(l) for l in lines if l.strip()]
        return results[:max_results]
    except FileNotFoundError:
        print('snscrape not installed or not in PATH. Install via pip: pip install snscrape')
        return []


def map_to_post(item, district):
    # Basic mapping from snscrape tweet JSON to our post shape
    post = {
        'district': district,
        'text': item.get('content') or item.get('rawContent') or '',
        'timestamp': item.get('date') or datetime.now().isoformat(),
        'engagement': {
            'likes': item.get('likeCount', 0),
            'retweets': item.get('retweetCount', 0) if 'retweetCount' in item else 0,
            'replies': item.get('replyCount', 0) if 'replyCount' in item else 0
        },
        'source': 'twitter_snscrape',
        'collected_at': datetime.now().isoformat()
    }
    return post


def post_batch(posts):
    try:
        r = requests.post(INGEST_URL, json=posts, timeout=30)
        print('POST', r.status_code, r.text[:200])
    except Exception as e:
        print('Failed to POST batch:', e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    parser.add_argument('--district', required=True)
    parser.add_argument('--max', type=int, default=200)
    parser.add_argument('--batch', type=int, default=20)
    args = parser.parse_args()

    items = run_snscrape(args.query, max_results=args.max)
    if not items:
        print('No items scraped')
        return

    batch = []
    for i, it in enumerate(items):
        post = map_to_post(it, args.district)
        batch.append(post)
        if len(batch) >= args.batch:
            post_batch(batch)
            batch = []
    if batch:
        post_batch(batch)


if __name__ == '__main__':
    main()
