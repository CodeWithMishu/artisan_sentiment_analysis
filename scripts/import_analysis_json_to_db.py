import json
import os
from datetime import datetime
from pymongo import MongoClient
from config import APIConfig

# Path to your analysis results JSON file(s)
DATA_DIR = 'data'

# Find all analysis_results_*.json files
json_files = [f for f in os.listdir(DATA_DIR) if f.startswith('analysis_results_') and f.endswith('.json')]

config = APIConfig()
client = MongoClient(config.mongodb_uri)
db = client[config.database_name]
sentiment_collection = db.sentiment_analysis

for filename in json_files:
    filepath = os.path.join(DATA_DIR, filename)
    print(f'Importing {filepath}...')
    with open(filepath, 'r', encoding='utf-8') as f:
        results = json.load(f)
        for district, result in results.items():
            sentiment_collection.replace_one(
                {'district': district},
                {
                    'district': district,
                    'analysis_date': result.get('timestamp', datetime.now()),
                    'analysis_results': result.get('analysis_results', result)
                },
                upsert=True
            )
    print(f'✅ Imported data for {len(results)} districts from {filename}')

print('🎉 All analysis results imported to database!')
