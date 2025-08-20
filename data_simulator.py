import json
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import os
from config import LocationConfig

class HPDataSimulator:
    def __init__(self):
        self.location_config = LocationConfig()
        
        # HP-specific content templates
        self.positive_templates = [
            "Beautiful handmade {product} from {district}! Amazing craftsmanship 😍",
            "Just bought authentic {product} from Himachal Pradesh. Love the quality!",
            "The traditional {product} from {district} is absolutely stunning",
            "Himachali {product} makes perfect gifts. Highly recommend!",
            "Supporting local artisans by buying {product} from HP",
            "Festival shopping done! Got beautiful {product} from {district}",
            "Export quality {product} from Himachal Pradesh artisans",
            "These {product} reflect rich culture of {district}",
            "Sustainable and eco-friendly {product} from HP hills",
            "Wedding decoration with traditional {product} from {district}"
        ]
        
        self.negative_templates = [
            "Disappointed with {product} quality from {district}",
            "Overpriced {product}, not worth the money",
            "Poor finishing on {product} from {district}",
            "Expected better quality in {product}",
            "Delivery delayed for {product} order",
            "Colors fading in {product} after few days",
            "Size doesn't match description for {product}",
            "Customer service poor for {product} queries"
        ]
        
        self.neutral_templates = [
            "Looking for authentic {product} from {district}",
            "Where can I find quality {product} in HP?",
            "Planning to visit {district} for {product} shopping",
            "Comparing prices for {product} from different districts",
            "Anyone knows about {product} workshops in {district}?",
            "Bulk order requirement for {product}",
            "Shipping options for {product} to Delhi?",
            "Custom design available for {product}?"
        ]
        
        # Product-specific keywords
        self.product_keywords = {
            'miniature_painting': ['kangra painting', 'pahari art', 'miniature art', 'traditional painting'],
            'chamba_rumal': ['chamba embroidery', 'silk embroidery', 'rumal', 'handwork'],
            'metal_craft': ['brass items', 'copper craft', 'metalwork', 'utensils'],
            'jewelry': ['silver jewelry', 'traditional ornaments', 'himachali jewelry'],
            'stone_carving': ['slate work', 'stone sculpture', 'carving', 'stonework'],
            'wooden_crafts': ['wood carving', 'wooden items', 'timber craft'],
            'pottery': ['clay items', 'pottery', 'ceramic work'],
            'textile_crafts': ['kullu shawl', 'handloom', 'weaving', 'woolen items'],
            'pine_needle_craft': ['pine craft', 'needle work', 'basket'],
            'herbal_products': ['himalayan herbs', 'medicinal plants', 'organic']
        }
    
    def generate_district_posts(self, district: str, categories: List[str], 
                              posts_per_category: int = 50) -> List[Dict]:
        """Generate simulated social media posts for specific district"""
        
        all_posts = []
        district_info = self.location_config.HP_DISTRICTS.get(district)
        
        if not district_info:
            print(f"Warning: District {district} not found in configuration")
            return []
        
        for category in categories:
            category_posts = self._generate_category_posts(
                district, category, posts_per_category
            )
            all_posts.extend(category_posts)
        
        return all_posts
    
    def _generate_category_posts(self, district: str, category: str, count: int) -> List[Dict]:
        """Generate posts for specific product category"""
        
        posts = []
        keywords = self.product_keywords.get(category, [category])
        
        for i in range(count):
            # Determine sentiment (60% positive, 25% neutral, 15% negative)
            sentiment_type = np.random.choice(
                ['positive', 'neutral', 'negative'], 
                p=[0.6, 0.25, 0.15]
            )
            
            # Select template and product
            if sentiment_type == 'positive':
                template = random.choice(self.positive_templates)
            elif sentiment_type == 'negative':
                template = random.choice(self.negative_templates)
            else:
                template = random.choice(self.neutral_templates)
            
            product = random.choice(keywords)
            
            # Generate post text
            post_text = template.format(
                product=product,
                district=district.replace('_', ' ').title()
            )
            
            # Add realistic variations
            post_text = self._add_variations(post_text)
            
            # Generate engagement metrics
            engagement = self._generate_engagement(sentiment_type)
            
            # Create post object
            post = {
                'platform': random.choice(['twitter', 'instagram', 'facebook']),
                'post_id': f"sim_{district}_{category}_{i}_{random.randint(1000, 9999)}",
                'text': post_text,
                'timestamp': self._generate_timestamp(),
                'engagement': engagement,
                'keyword': product,
                'category': category,
                'district': district,
                'location': {
                    'district': district,
                    'coordinates': self._add_location_noise(
                        self.location_config.HP_DISTRICTS[district]['center']
                    )
                },
                'sentiment_hint': sentiment_type,  # For validation
                'simulated': True,
                'collected_at': datetime.now()
            }
            
            posts.append(post)
        
        return posts
    
    def _add_variations(self, text: str) -> str:
        """Add realistic variations to text"""
        
        # Add emojis occasionally
        if random.random() < 0.3:
            emojis = ['👌', '🔥', '❤️', '👍', '🙌', '✨', '🎨', '🏔️']
            text += ' ' + random.choice(emojis)
        
        # Add hashtags occasionally
        if random.random() < 0.4:
            hashtags = ['#HimachalPradesh', '#Handmade', '#Artisan', '#Traditional', '#Craft']
            text += ' ' + random.choice(hashtags)
        
        return text
    
    def _generate_engagement(self, sentiment_type: str) -> Dict:
        """Generate realistic engagement metrics"""
        
        base_engagement = {
            'positive': {'likes': (10, 100), 'shares': (2, 20), 'comments': (1, 15)},
            'neutral': {'likes': (5, 30), 'shares': (0, 5), 'comments': (0, 5)},
            'negative': {'likes': (0, 10), 'shares': (0, 3), 'comments': (1, 8)}
        }
        
        ranges = base_engagement[sentiment_type]
        
        return {
            'likes': random.randint(*ranges['likes']),
            'retweets': random.randint(*ranges['shares']),  # Twitter terminology
            'replies': random.randint(*ranges['comments'])
        }
    
    def _generate_timestamp(self) -> datetime:
        """Generate realistic timestamp within last 30 days"""
        
        days_back = random.randint(0, 30)
        hours_back = random.randint(0, 23)
        minutes_back = random.randint(0, 59)
        
        return datetime.now() - timedelta(
            days=days_back, 
            hours=hours_back, 
            minutes=minutes_back
        )
    
    def _add_location_noise(self, coordinates: List[float]) -> List[float]:
        """Add small random noise to coordinates for realism"""
        
        # Add noise within ~5km radius
        noise_lat = random.uniform(-0.05, 0.05)
        noise_lon = random.uniform(-0.05, 0.05)
        
        return [
            coordinates[0] + noise_lon,  # longitude
            coordinates[1] + noise_lat   # latitude
        ]
    
    def generate_all_districts_data(self, posts_per_category: int = 30) -> Dict:
        """Generate data for all HP districts"""
        
        all_data = {}
        
        for district, info in self.location_config.HP_DISTRICTS.items():
            print(f"Generating data for {district}...")
            
            categories = info['artisan_specialties']
            district_posts = self.generate_district_posts(
                district, categories, posts_per_category
            )
            
            all_data[district] = district_posts
            print(f"Generated {len(district_posts)} posts for {district}")
        
        return all_data
    
    def save_simulated_data(self, data: Dict, filename: str = 'data/simulated_posts.json'):
        """Save simulated data to JSON file"""
        
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        serializable_data = {}
        for district, posts in data.items():
            serializable_posts = []
            for post in posts:
                serializable_post = post.copy()
                serializable_post['timestamp'] = post['timestamp'].isoformat()
                serializable_post['collected_at'] = post['collected_at'].isoformat()
                serializable_posts.append(serializable_post)
            serializable_data[district] = serializable_posts
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"Simulated data saved to {filename}")
    
    def load_simulated_data(self, filename: str = 'data/simulated_posts.json') -> Dict:
        """Load simulated data from JSON file"""
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert string timestamps back to datetime objects
            for district, posts in data.items():
                for post in posts:
                    post['timestamp'] = datetime.fromisoformat(post['timestamp'])
                    post['collected_at'] = datetime.fromisoformat(post['collected_at'])
            
            return data
        
        except FileNotFoundError:
            print(f"Simulated data file {filename} not found. Generate new data.")
            return {}
    
    def generate_trending_events(self) -> List[Dict]:
        """Generate trending events/festivals affecting demand"""
        
        events = [
            {
                'name': 'Dussehra Festival',
                'date': datetime.now() + timedelta(days=random.randint(0, 60)),
                'impact_categories': ['jewelry', 'textile_crafts', 'metal_craft'],
                'demand_multiplier': 2.5
            },
            {
                'name': 'Wedding Season',
                'date': datetime.now() + timedelta(days=random.randint(0, 90)),
                'impact_categories': ['jewelry', 'textile_crafts', 'chamba_rumal'],
                'demand_multiplier': 3.0
            },
            {
                'name': 'Tourism Peak Season',
                'date': datetime.now() + timedelta(days=random.randint(0, 120)),
                'impact_categories': ['miniature_painting', 'wooden_crafts', 'stone_carving'],
                'demand_multiplier': 2.0
            }
        ]
        
        return events

def main():
    """Generate simulated data for all districts"""
    
    print("🎯 HP ARTISAN DATA SIMULATOR")
    print("=" * 40)
    
    simulator = HPDataSimulator()
    
    # Generate data for all districts
    all_data = simulator.generate_all_districts_data(posts_per_category=50)
    
    # Save to file
    simulator.save_simulated_data(all_data)
    
    # Generate summary
    total_posts = sum(len(posts) for posts in all_data.values())
    print(f"\n📊 GENERATION COMPLETE")
    print(f"Total Districts: {len(all_data)}")
    print(f"Total Posts: {total_posts}")
    print(f"Average Posts per District: {total_posts // len(all_data)}")
    
    # Show sample posts
    print(f"\n📝 SAMPLE POSTS:")
    sample_district = list(all_data.keys())[0]
    for i, post in enumerate(all_data[sample_district][:3]):
        print(f"{i+1}. {post['text']}")
        print(f"   Engagement: {post['engagement']}")
        print(f"   Category: {post['category']}")
        print()

if __name__ == "__main__":
    main()
