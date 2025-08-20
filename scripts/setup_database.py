from pymongo import MongoClient
import json
from datetime import datetime
import os
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import APIConfig

def setup_database():
    """Initialize MongoDB database with required collections and indexes"""
    
    print("🗄️ Setting up HP Artisan Intelligence Database...")
    
    try:
        # Connect to MongoDB
        config = APIConfig()

        def _get_cfg_attr(cfg, name, default=None):
            if isinstance(cfg, dict):
                return cfg.get(name, default)
            return getattr(cfg, name, default)

        mongodb_uri = _get_cfg_attr(config, 'mongodb_uri', 'mongodb://localhost:27017/')
        database_name = _get_cfg_attr(config, 'database_name', 'hp_artisan_intelligence')

        client = MongoClient(mongodb_uri)
        db = client[database_name]
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise
    
    # Create collections
    collections = [
        'social_posts',
        'sentiment_analysis', 
        'demand_predictions',
        'audit_logs',
        'district_configs',
        'reports'
    ]
    
    for collection_name in collections:
        if collection_name not in db.list_collection_names():
            db.create_collection(collection_name)
            print(f"✅ Created collection: {collection_name}")
    
    # Create indexes for performance
    print("\n📊 Creating database indexes...")
    
    # Social posts indexes
    db.social_posts.create_index([("timestamp", -1)])
    db.social_posts.create_index([("district", 1)])
    db.social_posts.create_index([("category", 1)])
    db.social_posts.create_index([("platform", 1)])
    print("✅ Created indexes for social_posts")
    
    # Sentiment analysis indexes
    db.sentiment_analysis.create_index([("district", 1), ("analysis_date", -1)])
    db.sentiment_analysis.create_index([("analysis_date", -1)])
    print("✅ Created indexes for sentiment_analysis")
    
    # Demand predictions indexes
    db.demand_predictions.create_index([("district", 1), ("prediction_date", -1)])
    db.demand_predictions.create_index([("category", 1)])
    print("✅ Created indexes for demand_predictions")
    
    # Audit logs indexes
    db.audit_logs.create_index([("timestamp", -1)])
    db.audit_logs.create_index([("action", 1)])
    print("✅ Created indexes for audit_logs")
    
    # Insert initial configuration data
    print("\n⚙️ Inserting configuration data...")
    
    # District configurations
    from config import LocationConfig
    location_config = LocationConfig()
    
    for district, config in location_config.HP_DISTRICTS.items():
        district_doc = {
            'district': district,
            'config': config,
            'created_at': datetime.now(),
            'status': 'active'
        }
        
        db.district_configs.replace_one(
            {'district': district}, 
            district_doc, 
            upsert=True
        )
    
    print(f"✅ Inserted configuration for {len(location_config.HP_DISTRICTS)} districts")
    
    # Create initial audit log entry
    db.audit_logs.insert_one({
        'action': 'database_setup',
        'timestamp': datetime.now(),
        'details': 'Database initialized successfully',
        'status': 'success'
    })
    
    print("\n🎉 Database setup completed successfully!")
    # Print summary using safe accessors
    try:
        db_name = _get_cfg_attr(config, 'database_name', database_name)
        uri = _get_cfg_attr(config, 'mongodb_uri', mongodb_uri)
    except Exception:
        db_name = database_name
        uri = mongodb_uri

    print(f"Database: {db_name}")
    print(f"Collections: {len(collections)}")
    print(f"URI: {uri}")

def verify_database():
    """Verify database setup"""
    
    print("\n🔍 Verifying database setup...")
    
    config = APIConfig()
    client = MongoClient(config.mongodb_uri)
    db = client[config.database_name]
    
    # Check collections
    collections = db.list_collection_names()
    expected_collections = [
        'social_posts', 'sentiment_analysis', 'demand_predictions',
        'audit_logs', 'district_configs', 'reports'
    ]
    
    for collection in expected_collections:
        if collection in collections:
            count = db[collection].count_documents({})
            print(f"✅ {collection}: {count} documents")
        else:
            print(f"❌ {collection}: Missing")
    
    # Check indexes
    social_posts_indexes = db.social_posts.list_indexes()
    index_count = len(list(social_posts_indexes))
    print(f"✅ Indexes on social_posts: {index_count}")
    
    print("\n✅ Database verification complete!")

if __name__ == "__main__":
    setup_database()
    verify_database()
