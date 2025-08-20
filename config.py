import os
from dataclasses import dataclass
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class APIConfig:
    """API Configuration Settings"""
    # Twitter API v2 Configuration
    twitter_bearer_token: Optional[str] = os.getenv('TWITTER_BEARER_TOKEN')
    twitter_api_key: Optional[str] = os.getenv('TWITTER_API_KEY')
    twitter_api_secret: Optional[str] = os.getenv('TWITTER_API_SECRET')
    twitter_access_token: Optional[str] = os.getenv('TWITTER_ACCESS_TOKEN')
    twitter_access_token_secret: Optional[str] = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
    # Instagram Basic Display API
    instagram_access_token: Optional[str] = os.getenv('INSTAGRAM_ACCESS_TOKEN')
    
    # Facebook Graph API
    facebook_access_token: Optional[str] = os.getenv('FACEBOOK_ACCESS_TOKEN')
    
    # Database Configuration
    mongodb_uri: str = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    database_name: str = os.getenv('DATABASE_NAME', 'hp_artisan_intelligence')

@dataclass
class LocationConfig:
    """Himachal Pradesh Districts Configuration"""
    
    # Himachal Pradesh Districts with Coordinates
    HP_DISTRICTS = {
        'kangra': {
            'center': [76.2673, 32.0998],
            'radius': '50km',
            'artisan_specialties': ['miniature_painting', 'metal_craft', 'jewelry', 'stone_carving']
        },
        'chamba': {
            'center': [76.1247, 32.5522],
            'radius': '40km', 
            'artisan_specialties': ['chamba_rumal', 'metal_craft', 'jewelry']
        },
        'solan': {
            'center': [77.0999, 30.9045],
            'radius': '35km',
            'artisan_specialties': ['metal_craft', 'jewelry', 'stone_carving']
        },
        'mandi': {
            'center': [76.9318, 31.7058],
            'radius': '45km',
            'artisan_specialties': ['jewelry', 'stone_carving']
        },
        'bilaspur': {
            'center': [76.7553, 31.3314],
            'radius': '30km',
            'artisan_specialties': ['stone_carving']
        },
        'sirmaur': {
            'center': [77.2910, 30.5629],
            'radius': '40km',
            'artisan_specialties': ['metal_craft']
        },
        'shimla': {
            'center': [77.1734, 31.1048],
            'radius': '50km',
            'artisan_specialties': ['metal_craft']
        },
        'kinnaur': {
            'center': [78.2353, 31.6077],
            'radius': '60km',
            'artisan_specialties': ['metal_craft', 'jewelry']
        },
        'lahaul_spiti': {
            'center': [77.6131, 32.5734],
            'radius': '70km',
            'artisan_specialties': ['jewelry']
        },
        'una': {
            'center': [76.2711, 31.4685],
            'radius': '25km',
            'artisan_specialties': ['metal_craft']
        }
    }
    
    # Artisan Product Categories (Based on HP Government Data)
    ARTISAN_CATEGORIES = {
        'chamba_rumal': ['chamba rumal', 'chamba embroidery', 'himachali embroidery'],
        'miniature_painting': ['kangra painting', 'pahari painting', 'miniature art'],
        'metal_craft': ['himachali metalwork', 'brass items', 'copper craft'],
        'jewelry': ['himachali jewelry', 'traditional ornaments', 'silver jewelry'],
        'stone_carving': ['stone craft', 'slate carving', 'sculpture'],
        'wooden_crafts': ['wood carving', 'wooden artifacts', 'timber craft'],
        'pottery': ['himachali pottery', 'clay items', 'ceramic craft'],
        'textile_crafts': ['handloom', 'weaving', 'kullu shawl', 'kinnauri shawl'],
        'pine_needle_craft': ['pine needle baskets', 'pine craft'],
        'herbal_products': ['himalayan herbs', 'medicinal plants', 'forest products']
    }

# Government Compliance Settings
@dataclass
class ComplianceConfig:
    """Government Compliance Configuration"""
    data_retention_days: int = int(os.getenv('REPORT_RETENTION_DAYS', '365'))
    privacy_mode: bool = os.getenv('PRIVACY_MODE', 'True').lower() == 'true'
    audit_logging: bool = os.getenv('AUDIT_LOGGING', 'True').lower() == 'true'
    encrypted_storage: bool = os.getenv('ENCRYPTED_STORAGE', 'True').lower() == 'true'
    gdpr_compliant: bool = True
    report_template: str = "government_standard"
