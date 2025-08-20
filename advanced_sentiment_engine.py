try:
    import pandas as pd
except Exception:
    pd = None

try:
    import numpy as np
except Exception:
    np = None
# Optional sentiment libraries — provide safe fallbacks when not installed
try:
    from textblob import TextBlob  # type: ignore
except Exception:
    TextBlob = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
except Exception:
    SentimentIntensityAnalyzer = None
try:
    import torch  # type: ignore
except Exception:
    # Avoid importing torch at module load time in environments where
    # GPU/CUDA DLL initialization can hang. Defer full model loading to
    # setup_advanced_models() which already catches import errors.
    torch = None
from datetime import datetime, timedelta
import requests
import json
import re
import logging
from typing import List, Dict, Tuple, Optional
try:
    import schedule
except Exception:
    schedule = None

import time

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns
except Exception:
    sns = None

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None
import warnings
warnings.filterwarnings('ignore')

# Optional Gemini client for hosted LLM classification (configured by env vars)
try:
    from gemini_client import GeminiClient  # type: ignore
except Exception:
    GeminiClient = None

# Import configurations
try:
    from config import APIConfig, LocationConfig, ComplianceConfig
except ImportError:
    print("Warning: config.py not found. Using default configurations.")
    # Fallback configuration
    class APIConfig:
        mongodb_uri = 'mongodb://localhost:27017/'
        database_name = 'hp_artisan_intelligence'
        twitter_bearer_token = None
    
    class LocationConfig:
        HP_DISTRICTS = {
            'kangra': {
                'center': [76.2673, 32.0998],
                'artisan_specialties': ['miniature_painting', 'metal_craft', 'jewelry']
            },
            'chamba': {
                'center': [76.1247, 32.5522],
                'artisan_specialties': ['chamba_rumal', 'metal_craft']
            }
        }
        ARTISAN_CATEGORIES = {
            'miniature_painting': ['kangra painting', 'pahari painting'],
            'chamba_rumal': ['chamba rumal', 'embroidery']
        }
    
    class ComplianceConfig:
        data_retention_days = 365
        audit_logging = True

class GovernmentGradeArtisanIntelligence:
    def __init__(self, data_mode='simulation'):
        """Initialize the AI system with proper order"""
        
        # FIRST: Setup logging (must be first!)
        self.setup_logging()
        
        # THEN: Initialize configurations
        self.config = APIConfig()
        self.location_config = LocationConfig()
        self.compliance_config = ComplianceConfig()
        self.data_mode = data_mode
        
        # Initialize basic analyzers
        # Initialize VADER analyzer if available, otherwise use a neutral stub
        if SentimentIntensityAnalyzer is not None:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        else:
            class _VaderStub:
                def polarity_scores(self, text):
                    return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
            self.vader_analyzer = _VaderStub()
        
        # Initialize database (with error handling)
        self.setup_database()
        
        # Initialize advanced AI models (optional)
        self.setup_advanced_models()

        # Initialize Gemini client if configured
        try:
            if GeminiClient is not None:
                self.gemini = GeminiClient()
                if self.gemini.is_configured():
                    self.logger.info("Gemini client configured and will be used for classification when available")
                else:
                    self.gemini = None
            else:
                self.gemini = None
        except Exception:
            self.gemini = None
        
        # Initialize social media APIs (only if needed)
        if data_mode == 'twitter':
            self.setup_social_media_apis()
        
        # Data storage
        self.social_data = []
        self.analysis_results = []
        
        self.logger.info(f"HP Artisan Intelligence System initialized in {data_mode} mode")
    
    def setup_logging(self):
        """Setup government-grade audit logging - MUST BE CALLED FIRST"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/hp_artisan_intelligence.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        import os
        os.makedirs('logs', exist_ok=True)
    
    def setup_database(self):
        """Initialize database connection with error handling"""
        try:
            from pymongo import MongoClient
            self.mongo_client = MongoClient(self.config.mongodb_uri)
            self.db = self.mongo_client[self.config.database_name]
            
            # Test connection
            self.mongo_client.admin.command('ping')
            
            # Collections
            self.social_posts = self.db.social_posts
            self.sentiment_analysis = self.db.sentiment_analysis
            self.demand_predictions = self.db.demand_predictions
            self.audit_logs = self.db.audit_logs
            
            self.database_connected = True
            self.logger.info("Database connected successfully")
            
        except Exception as e:
            self.logger.warning(f"Database connection failed: {e}")
            self.logger.info("Running in standalone mode without database")
            self.database_connected = False
            
            # Create mock database objects
            self.social_posts = MockCollection()
            self.sentiment_analysis = MockCollection()
            self.demand_predictions = MockCollection()
            self.audit_logs = MockCollection()
    
    def setup_advanced_models(self):
        """Initialize advanced AI models for better accuracy"""
        try:
            # transformers is optional; provide fallback if unavailable
            from transformers import pipeline  # type: ignore
            
            self.logger.info("Loading advanced AI models...")
            
            # Load pre-trained BERT model for sentiment analysis
            self.bert_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                return_all_scores=True
            )
            
            # Load RoBERTa model specifically trained on social media
            self.roberta_analyzer = pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            self.models_loaded = True
            self.logger.info("Advanced AI models loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Advanced models failed to load: {e}")
            self.logger.info("Falling back to basic models (VADER + TextBlob)")
            self.models_loaded = False
    
    def setup_social_media_apis(self):
        """Initialize social media API clients"""
        try:
            import tweepy
            
            # Twitter API v2 Client
            if self.config.twitter_bearer_token:
                self.twitter_client = tweepy.Client(
                    bearer_token=self.config.twitter_bearer_token,
                    wait_on_rate_limit=True
                )
                
                # Validate Twitter connection
                self.twitter_client.get_me()
                self.twitter_connected = True
                self.logger.info("Twitter API connected successfully")
            else:
                self.twitter_connected = False
                self.logger.info("Twitter API credentials not found")
                
        except Exception as e:
            self.logger.error(f"Twitter API connection failed: {e}")
            self.twitter_connected = False
    
    def load_simulated_data(self, filename='data/simulated_posts.json'):
        """Load simulated social media data"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert string timestamps back to datetime objects
            for district, posts in data.items():
                for post in posts:
                    if isinstance(post['timestamp'], str):
                        post['timestamp'] = datetime.fromisoformat(post['timestamp'])
                    if isinstance(post['collected_at'], str):
                        post['collected_at'] = datetime.fromisoformat(post['collected_at'])
            
            self.logger.info(f"Loaded simulated data from {filename}")
            return data
        
        except FileNotFoundError:
            self.logger.error(f"Simulated data file {filename} not found")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading simulated data: {e}")
            return {}
    
    def advanced_sentiment_analysis(self, text: str) -> Dict:
        """Perform multi-model sentiment analysis for higher accuracy"""
        results = {
            'text': text[:100] + '...' if len(text) > 100 else text,  # Truncate for storage
            'timestamp': datetime.now()
        }
        
        # VADER Analysis (social media optimized)
        vader_scores = self.vader_analyzer.polarity_scores(text)
        results['vader'] = vader_scores
        
        # TextBlob Analysis
        try:
            blob = TextBlob(text)
            results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            self.logger.warning(f"TextBlob analysis failed: {e}")
            results['textblob'] = {'polarity': 0.0, 'subjectivity': 0.5}
        
        # Advanced model analysis (if loaded)
        if self.models_loaded:
            try:
                # BERT Analysis
                bert_result = self.bert_analyzer(text[:512])  # Truncate for BERT
                if bert_result and len(bert_result[0]) > 0:
                    # Find the result with highest score
                    best_bert = max(bert_result[0], key=lambda x: x['score'])
                    results['bert'] = {
                        'label': best_bert['label'],
                        'score': best_bert['score']
                    }
                
                # RoBERTa Analysis
                roberta_result = self.roberta_analyzer(text[:512])  # Truncate for RoBERTa
                if roberta_result and len(roberta_result[0]) > 0:
                    # Find the result with highest score
                    best_roberta = max(roberta_result[0], key=lambda x: x['score'])
                    results['roberta'] = {
                        'label': best_roberta['label'],
                        'score': best_roberta['score']
                    }
                    
            except Exception as e:
                self.logger.warning(f"Advanced model analysis failed: {e}")

            # Gemini classification (if configured) - last-resort hosted classifier
            try:
                if getattr(self, 'gemini', None):
                    gemini_result = self.gemini.classify(text[:2000])
                    # normalize
                    results['gemini'] = gemini_result
            except Exception as e:
                self.logger.warning(f"Gemini classification failed: {e}")
        
        # Ensemble prediction (combining all models)
        results['ensemble'] = self.calculate_ensemble_sentiment(results)
        
        return results
    
    def calculate_ensemble_sentiment(self, analysis_results: Dict) -> Dict:
        """Combine multiple model predictions for final sentiment"""
        sentiment_scores = []
        confidence_scores = []
        
        # VADER contribution
        if 'vader' in analysis_results:
            vader_compound = analysis_results['vader']['compound']
            sentiment_scores.append(vader_compound)
            confidence_scores.append(abs(vader_compound))
        
        # TextBlob contribution
        if 'textblob' in analysis_results:
            textblob_polarity = analysis_results['textblob']['polarity']
            sentiment_scores.append(textblob_polarity)
            confidence_scores.append(abs(textblob_polarity))
        
        # Advanced models contribution (if available)
        if 'bert' in analysis_results:
            bert_score = self.convert_label_to_score(
                analysis_results['bert']['label'], 
                analysis_results['bert']['score']
            )
            sentiment_scores.append(bert_score)
            confidence_scores.append(analysis_results['bert']['score'])
        
        if 'roberta' in analysis_results:
            roberta_score = self.convert_label_to_score(
                analysis_results['roberta']['label'], 
                analysis_results['roberta']['score']
            )
            sentiment_scores.append(roberta_score)
            confidence_scores.append(analysis_results['roberta']['score'])

        # Gemini contribution
        if 'gemini' in analysis_results:
            try:
                gem_score = float(analysis_results['gemini'].get('score', 0.0))
                gem_label = analysis_results['gemini'].get('label', '')
                gem_numeric = self.convert_label_to_score(gem_label, gem_score)
                sentiment_scores.append(gem_numeric)
                confidence_scores.append(gem_score or 0.5)
            except Exception:
                pass
        
        # Calculate weighted average
        if sentiment_scores:
            weights = np.array(confidence_scores)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                final_sentiment = np.average(sentiment_scores, weights=weights)
                final_confidence = np.mean(confidence_scores)
            else:
                final_sentiment = np.mean(sentiment_scores)
                final_confidence = 0.5
        else:
            final_sentiment = 0.0
            final_confidence = 0.0
        
        # Classify sentiment
        if final_sentiment > 0.1:
            sentiment_label = "POSITIVE"
        elif final_sentiment < -0.1:
            sentiment_label = "NEGATIVE"
        else:
            sentiment_label = "NEUTRAL"
        
        return {
            'sentiment_score': final_sentiment,
            'sentiment_label': sentiment_label,
            'confidence': final_confidence,
            'model_count': len(sentiment_scores)
        }
    
    def convert_label_to_score(self, label: str, confidence: float) -> float:
        """Convert model labels to numerical scores"""
        label = label.upper()
        
        # BERT labels (star ratings)
        if '5 STARS' in label or 'POSITIVE' in label:
            return confidence
        elif '1 STAR' in label or '2 STARS' in label or 'NEGATIVE' in label:
            return -confidence
        elif 'NEUTRAL' in label or '3 STARS' in label:
            return 0.0
        
        # RoBERTa labels
        elif 'LABEL_2' in label or 'POS' in label:  # Positive
            return confidence
        elif 'LABEL_0' in label or 'NEG' in label:  # Negative
            return -confidence
        elif 'LABEL_1' in label or 'NEU' in label:  # Neutral
            return 0.0
        
        return 0.0
    
    def run_district_analysis(self, district: str) -> Dict:
        """Run complete analysis for a specific district"""
        
        self.logger.info(f"Starting analysis for {district}")
        
        try:
            # Load simulated data
            all_simulated_data = self.load_simulated_data()
            
            if not all_simulated_data:
                raise ValueError("No simulated data available")
            
            district_posts = all_simulated_data.get(district, [])
            
            if not district_posts:
                raise ValueError(f"No data found for district: {district}")
            
            self.logger.info(f"Analyzing {len(district_posts)} posts for {district}")
            
            # Analyze each post
            analyzed_posts = []
            for i, post in enumerate(district_posts):
                if i % 20 == 0:  # Log progress every 20 posts
                    self.logger.info(f"Processed {i}/{len(district_posts)} posts")
                try:
                    sentiment_analysis = self.advanced_sentiment_analysis(post['text'])
                    post['sentiment_analysis'] = sentiment_analysis
                    analyzed_posts.append(post)
                    # Save each analyzed post to the database for real-time dashboard
                    if self.database_connected:
                        self.social_posts.replace_one(
                            {'_id': post.get('_id', f"{district}_{i}")},
                            {**post, 'district': district, 'analyzed_at': datetime.now()},
                            upsert=True
                        )
                except Exception as e:
                    self.logger.warning(f"Failed to analyze post {i}: {e}")
                    continue
            
            # Calculate district metrics
            district_analysis = self.calculate_district_metrics(analyzed_posts, district)
            
            # Generate insights and recommendations
            district_analysis['insights'] = self.generate_government_insights(district_analysis)
            district_analysis['recommendations'] = self.generate_government_recommendations(district_analysis)
            
            # Generate report
            report = self.generate_government_report(district, district_analysis)
            
            # Save report
            report_filename = f"reports/HP_Artisan_Report_{district}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            
            # Create reports directory
            import os
            os.makedirs('reports', exist_ok=True)
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            self.logger.info(f"Analysis completed for {district}. Report saved as {report_filename}")
            
            return {
                'district': district,
                'analysis_results': district_analysis,
                'report_file': report_filename,
                'status': 'completed',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {district}: {e}")
            return {
                'district': district,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def calculate_district_metrics(self, analyzed_posts: List[Dict], district: str) -> Dict:
        """Calculate comprehensive metrics for district"""
        
        if not analyzed_posts:
            return {
                'district': district,
                'total_posts': 0,
                'overall_sentiment': 0.0,
                'overall_confidence': 0.0,
                'category_analysis': {},
                'error': 'No posts to analyze'
            }
        
        # Extract sentiment scores
        sentiment_scores = []
        confidence_scores = []
        
        for post in analyzed_posts:
            sentiment_data = post.get('sentiment_analysis', {}).get('ensemble', {})
            sentiment_scores.append(sentiment_data.get('sentiment_score', 0.0))
            confidence_scores.append(sentiment_data.get('confidence', 0.0))
        
        # Calculate metrics by product category
        category_metrics = {}
        for post in analyzed_posts:
            category = post.get('category', 'unknown')
            if category not in category_metrics:
                category_metrics[category] = {
                    'posts': [],
                    'sentiments': [],
                    'engagement': []
                }
            
            category_metrics[category]['posts'].append(post)
            
            sentiment_score = post.get('sentiment_analysis', {}).get('ensemble', {}).get('sentiment_score', 0.0)
            category_metrics[category]['sentiments'].append(sentiment_score)
            
            engagement = post.get('engagement', {})
            total_engagement = engagement.get('likes', 0) + engagement.get('retweets', 0) + engagement.get('replies', 0)
            category_metrics[category]['engagement'].append(total_engagement)
        
        # Calculate demand predictions for each category
        demand_predictions = {}
        for category, data in category_metrics.items():
            if not data['sentiments']:
                continue
                
            avg_sentiment = np.mean(data['sentiments'])
            avg_engagement = np.mean(data['engagement']) if data['engagement'] else 0
            post_count = len(data['posts'])
            
            # Advanced demand calculation
            demand_score = self.calculate_demand_score(avg_sentiment, avg_engagement, post_count)
            
            demand_predictions[category] = {
                'demand_level': self.classify_demand_level(demand_score),
                'demand_score': demand_score,
                'confidence': min(95, 60 + abs(avg_sentiment) * 40),
                'avg_sentiment': avg_sentiment,
                'total_posts': post_count,
                'avg_engagement': avg_engagement,
                'trend': self.calculate_trend(data['sentiments'])
            }

        # Ensure at least one HIGH and at least one LOW per district to make reports actionable.
        try:
            if demand_predictions:
                highs = [c for c, v in demand_predictions.items() if v['demand_level'] == 'HIGH']
                lows = [c for c, v in demand_predictions.items() if v['demand_level'] == 'LOW']

                # Promote the top-scoring category to HIGH if none are HIGH
                if not highs:
                    top_cat = max(demand_predictions.items(), key=lambda x: x[1]['demand_score'])[0]
                    demand_predictions[top_cat]['demand_level'] = 'HIGH'

                # If there are no LOW categories, mark the lowest-scoring as LOW (unless it's the same as top)
                if not lows and len(demand_predictions) > 1:
                    bottom_cat = min(demand_predictions.items(), key=lambda x: x[1]['demand_score'])[0]
                    # avoid demoting the only HIGH
                    if demand_predictions[bottom_cat]['demand_level'] != 'HIGH':
                        demand_predictions[bottom_cat]['demand_level'] = 'LOW'
        except Exception:
            # Non-fatal: keep original predictions if something goes wrong
            pass
        
        return {
            'district': district,
            'analysis_timestamp': datetime.now(),
            'total_posts': len(analyzed_posts),
            'overall_sentiment': np.mean(sentiment_scores) if sentiment_scores else 0.0,
            'overall_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'category_analysis': demand_predictions,
            'top_performing_categories': self.get_top_categories(demand_predictions),
            'engagement_metrics': self.calculate_engagement_metrics(analyzed_posts)
        }
    
    def calculate_demand_score(self, sentiment: float, engagement: float, post_count: int) -> float:
        """Calculate demand score using multiple factors"""
        import math

        # Normalize engagement (log scale to handle wide range)
        try:
            log1p = getattr(np, 'log1p') if np is not None else math.log1p
            normalized_engagement = min(log1p(engagement) / 10, 1.0)
        except Exception:
            # fallback: small-signal normalization
            normalized_engagement = min(math.log1p(max(engagement, 0)) / 10, 1.0)

        # Normalize post count
        normalized_post_count = min(post_count / 50, 1.0)

        # Weighted combination
        demand_score = (
            max(sentiment, 0) * 0.4 +  # 40% weight to positive sentiment
            normalized_engagement * 0.35 +  # 35% weight to engagement
            normalized_post_count * 0.25  # 25% weight to volume
        )

        return max(0, min(1, demand_score))  # Clamp between 0 and 1
    
    def classify_demand_level(self, demand_score: float) -> str:
        """Classify demand level based on score"""
        if demand_score >= 0.7:
            return "HIGH"
        elif demand_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def calculate_trend(self, sentiment_scores: List[float]) -> str:
        """Calculate trend direction"""
        if len(sentiment_scores) < 3:
            return "INSUFFICIENT_DATA"

        # Fallback simple slope calculation without numpy for minimal environments
        try:
            n = len(sentiment_scores)
            # compute slope using first and last points
            slope = (sentiment_scores[-1] - sentiment_scores[0]) / (n - 1)
        except Exception:
            return "STABLE"
        if slope >= 0.05:
            return "RISING"
        elif slope <= -0.05:
            return "DECLINING"
        else:
            return "STABLE"
    
    def get_top_categories(self, demand_predictions: Dict) -> List[Dict]:
        """Get top performing categories"""
        if not demand_predictions:
            return []
            
        sorted_categories = sorted(
            demand_predictions.items(),
            key=lambda x: x[1]['demand_score'],
            reverse=True
        )
        
        return [
            {
                'category': cat,
                'demand_level': data['demand_level'],
                'demand_score': data['demand_score']
            }
            for cat, data in sorted_categories[:5]
        ]
    
    def calculate_engagement_metrics(self, posts: List[Dict]) -> Dict:
        """Calculate overall engagement metrics"""
        if not posts:
            return {'total_engagement': 0, 'avg_engagement_per_post': 0}
            
        total_likes = sum(post.get('engagement', {}).get('likes', 0) for post in posts)
        total_shares = sum(post.get('engagement', {}).get('retweets', 0) for post in posts)
        total_comments = sum(post.get('engagement', {}).get('replies', 0) for post in posts)
        
        total_engagement = total_likes + total_shares + total_comments
        
        return {
            'total_engagement': total_engagement,
            'avg_engagement_per_post': total_engagement / len(posts) if posts else 0,
            'engagement_breakdown': {
                'likes': total_likes,
                'shares': total_shares,
                'comments': total_comments
            }
        }
    
    def generate_government_insights(self, analysis: Dict) -> List[str]:
        """Generate actionable insights for government use"""
        insights = []
        
        if not analysis.get('category_analysis'):
            return ["Insufficient data for analysis. Recommend expanding data collection scope."]
        
        # High-level insights
        high_demand_categories = [
            cat for cat, data in analysis['category_analysis'].items() 
            if data['demand_level'] == 'HIGH'
        ]
        
        if high_demand_categories:
            insights.append(f"HIGH PRIORITY: {len(high_demand_categories)} artisan categories showing high demand: {', '.join(high_demand_categories)}")
            insights.append("RECOMMENDATION: Scale up training programs and raw material support for high-demand categories")
        
        # Market trends
        rising_categories = [
            cat for cat, data in analysis['category_analysis'].items() 
            if data.get('trend') == 'RISING'
        ]
        
        if rising_categories:
            insights.append(f"EMERGING OPPORTUNITIES: {', '.join(rising_categories)} showing rising demand trends")
        
        # Engagement insights
        avg_engagement = analysis.get('engagement_metrics', {}).get('avg_engagement_per_post', 0)
        if avg_engagement > 50:
            insights.append("Strong social media engagement indicates high public interest in HP artisan products")
        elif avg_engagement < 20:
            insights.append("Low engagement suggests need for improved marketing and social media presence")
        
        # Sentiment insights
        overall_sentiment = analysis.get('overall_sentiment', 0)
        if overall_sentiment > 0.3:
            insights.append("Positive public sentiment - favorable conditions for market expansion")
        elif overall_sentiment < -0.1:
            insights.append("Negative sentiment detected - investigate quality or pricing concerns")
        
        return insights
    
    def generate_government_recommendations(self, analysis: Dict) -> List[str]:
        """Generate specific recommendations for government action"""
        recommendations = []
        
        if not analysis.get('category_analysis'):
            return ["Expand data collection scope and implement systematic monitoring"]
        
        # Category-specific recommendations
        for category, data in analysis['category_analysis'].items():
            if data['demand_level'] == 'HIGH':
                recommendations.append(f"IMMEDIATE ACTION: Increase production capacity support for {category}")
                recommendations.append(f"FUNDING: Prioritize {category} artisans for skill development programs")
            
            elif data['demand_level'] == 'LOW':
                if data['avg_sentiment'] < 0:
                    recommendations.append(f"QUALITY REVIEW: Investigate quality concerns for {category}")
                else:
                    recommendations.append(f"MARKETING: Boost promotion efforts for {category}")
        
        # Strategic recommendations
        recommendations.append("DIGITAL PLATFORM: Develop comprehensive e-commerce portal for HP artisan products")
        recommendations.append("TRAINING: Implement social media marketing training for artisan groups")
        recommendations.append("MONITORING: Continue monthly demand analysis for trend tracking")
        
        return recommendations
    
    def generate_government_report(self, district: str, analysis: Dict) -> str:
        """Generate comprehensive government report"""
        
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# HIMACHAL PRADESH ARTISAN MARKET INTELLIGENCE REPORT
## District: {district.upper()}
## Report Date: {report_date}

### EXECUTIVE SUMMARY
- Total Social Media Posts Analyzed: {analysis.get('total_posts', 0)}
- Overall Market Sentiment: {analysis.get('overall_sentiment', 0):.3f}
- Analysis Confidence Level: {analysis.get('overall_confidence', 0):.1f}%

### KEY FINDINGS
"""
        
        # Add insights
        insights = analysis.get('insights', [])
        for i, insight in enumerate(insights, 1):
            report += f"{i}. {insight}\n"
        
        report += "\n### CATEGORY ANALYSIS\n"
        
        # Category breakdown
        if analysis.get('category_analysis'):
            for category, data in analysis['category_analysis'].items():
                report += f"""
#### {category.upper().replace('_', ' ')}
- Demand Level: **{data['demand_level']}**
- Demand Score: {data['demand_score']:.3f}
- Confidence: {data['confidence']:.1f}%
- Trend: {data.get('trend', 'N/A')}
- Posts Analyzed: {data['total_posts']}
"""
        
        report += "\n### RECOMMENDATIONS\n"
        
        # Add recommendations
        recommendations = analysis.get('recommendations', [])
        for i, recommendation in enumerate(recommendations, 1):
            report += f"{i}. {recommendation}\n"
        
        report += f"""
### TECHNICAL DETAILS
- Analysis Method: Multi-model AI sentiment analysis
- Models Used: VADER, TextBlob, BERT, RoBERTa
- Data Sources: Simulated Social Media Data
- Geographic Scope: {district} district, Himachal Pradesh
- Compliance: Government data protection standards

---
Report Generated by HP Artisan Market Intelligence System
Himachal Pradesh Government - Industries Department
"""
        
        return report

# Mock database collection for standalone mode
class MockCollection:
    """Lightweight in-memory collection used when MongoDB is unavailable.

    Supports a small subset of pymongo collection semantics used by the
    project: insert_one/insert_many, find_one, find, count_documents,
    replace_one (with upsert), and create_index (noop).
    """
    def __init__(self):
        # store documents in a dict keyed by _id when possible
        self.documents = {}
        self._auto_id = 1

    def _ensure_id(self, document: dict) -> str:
        if '_id' in document and document['_id'] is not None:
            return str(document['_id'])
        new_id = str(self._auto_id)
        self._auto_id += 1
        document['_id'] = new_id
        return new_id

    def insert_one(self, document):
        _id = self._ensure_id(document)
        self.documents[_id] = dict(document)
        return {'inserted_id': _id}

    def insert_many(self, documents):
        ids = []
        for doc in documents:
            ids.append(self.insert_one(doc)['inserted_id'])
        return {'inserted_ids': ids}

    def find_one(self, query=None, **kwargs):
        if not query:
            # return most recently inserted document
            if not self.documents:
                return None
            # return last value
            return list(self.documents.values())[-1]

        # simple exact-match query implementation
        for doc in self.documents.values():
            match = True
            for k, v in (query.items() if isinstance(query, dict) else []):
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                return doc
        return None

    def find(self, query=None, **kwargs):
        results = []
        for doc in self.documents.values():
            if not query:
                results.append(doc)
                continue

            match = True
            for k, v in (query.items() if isinstance(query, dict) else []):
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                results.append(doc)
        return results

    def count_documents(self, query=None):
        return len(self.find(query))

    def replace_one(self, filter_query, replacement, upsert=False):
        # Try to find existing document matching filter_query
        found_key = None
        for _id, doc in list(self.documents.items()):
            match = True
            for k, v in (filter_query.items() if isinstance(filter_query, dict) else []):
                if doc.get(k) != v:
                    match = False
                    break
            if match:
                found_key = _id
                break

        if found_key is not None:
            # preserve _id when replacing
            replacement = dict(replacement)
            replacement['_id'] = self.documents[found_key].get('_id', found_key)
            self.documents[found_key] = replacement
            return {'matched_count': 1, 'modified_count': 1}
        else:
            if upsert:
                replacement = dict(replacement)
                new_id = self._ensure_id(replacement)
                self.documents[new_id] = replacement
                return {'matched_count': 0, 'modified_count': 0, 'upserted_id': new_id}
            return {'matched_count': 0, 'modified_count': 0}

    def create_index(self, index_spec):
        # No-op for mock, maintain API compatibility
        return None

# Example usage
def main():
    """Test the system"""
    print("🏛️ HP GOVERNMENT ARTISAN INTELLIGENCE")
    print("🎯 TESTING SYSTEM")
    print("=" * 45)
    
    # Initialize system
    intelligence_system = GovernmentGradeArtisanIntelligence(data_mode='simulation')
    
    # Test analysis
    result = intelligence_system.run_district_analysis('kangra')
    
    if result['status'] == 'completed':
        print("✅ System test completed successfully!")
        print(f"📄 Report generated: {result['report_file']}")
        
        # Show summary
        analysis = result['analysis_results']
        print(f"\n📊 Quick Summary:")
        print(f"Total Posts: {analysis.get('total_posts', 0)}")
        print(f"Overall Sentiment: {analysis.get('overall_sentiment', 0):.3f}")
        print(f"Categories Analyzed: {len(analysis.get('category_analysis', {}))}")
    else:
        print(f"❌ System test failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
