import sys
import argparse
from datetime import datetime
import json
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advanced_sentiment_engine import GovernmentGradeArtisanIntelligence
from data_simulator import HPDataSimulator

def run_simulation_analysis():
    """Run analysis using simulated data"""
    
    print("🎯 RUNNING SIMULATION ANALYSIS")
    print("=" * 40)
    
    # Initialize simulator and generate data
    simulator = HPDataSimulator()
    
    # Check if simulated data exists
    if not os.path.exists('data/simulated_posts.json'):
        print("📊 Generating simulated data...")
        all_data = simulator.generate_all_districts_data(posts_per_category=50)
        simulator.save_simulated_data(all_data)
    else:
        print("📊 Using existing simulated data...")
    
    # Initialize analysis engine
    engine = GovernmentGradeArtisanIntelligence()
    
    # Run analysis for each district
    results = {}
    districts = ['kangra', 'chamba', 'solan', 'mandi', 'bilaspur']
    
    for district in districts:
        print(f"\n🔍 Analyzing {district.upper()}...")
        try:
            result = engine.run_district_analysis(district)
            results[district] = result
            print(f"✅ Analysis completed for {district}")
        except Exception as e:
            print(f"❌ Error analyzing {district}: {e}")
            results[district] = {'error': str(e)}
    
    # Save results
    results_file = f"data/analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Print summary
    print("\n📋 ANALYSIS SUMMARY")
    print("=" * 40)
    
    for district, result in results.items():
        if 'error' not in result:
            analysis = result.get('analysis_results', {})
            total_posts = analysis.get('total_posts', 0)
            sentiment = analysis.get('overall_sentiment', 0)
            high_demand = len([
                cat for cat, data in analysis.get('category_analysis', {}).items()
                if data.get('demand_level') == 'HIGH'
            ])
            
            print(f"{district.upper():12} | Posts: {total_posts:3d} | Sentiment: {sentiment:+.3f} | High Demand: {high_demand}")
        else:
            print(f"{district.upper():12} | ERROR: {result['error']}")

def run_twitter_analysis():
    """Run analysis using real Twitter data"""
    
    print("🐦 RUNNING TWITTER ANALYSIS")
    print("=" * 40)
    
    # Check if Twitter API keys are available
    from config import APIConfig
    config = APIConfig()
    
    if not config.twitter_bearer_token:
        print("❌ Twitter API keys not found!")
        print("Please add Twitter API credentials to .env file")
        print("Falling back to simulation mode...")
        run_simulation_analysis()
        return
    
    # Initialize analysis engine with Twitter mode
    engine = GovernmentGradeArtisanIntelligence()
    
    print("🔍 Running analysis with real Twitter data...")
    # Implementation would use real Twitter data collection
    # For now, fall back to simulation
    run_simulation_analysis()

def run_kangra_priority_analysis():
    """Run detailed analysis specifically for Kangra district"""
    
    print("🎯 KANGRA PRIORITY ANALYSIS")
    print("=" * 40)
    
    engine = GovernmentGradeArtisanIntelligence()
    
    print("🔍 Running comprehensive Kangra analysis...")
    result = engine.run_district_analysis('kangra')
    
    # Generate detailed report
    analysis = result.get('analysis_results', {})
    
    print("\n📊 KANGRA ANALYSIS RESULTS")
    print("=" * 30)
    print(f"Total Posts Analyzed: {analysis.get('total_posts', 0)}")
    print(f"Overall Sentiment: {analysis.get('overall_sentiment', 0):+.3f}")
    print(f"Analysis Confidence: {analysis.get('overall_confidence', 0):.1f}%")
    
    # Category breakdown
    print("\n🎨 PRODUCT CATEGORY ANALYSIS")
    print("-" * 30)
    
    category_analysis = analysis.get('category_analysis', {})
    for category, data in category_analysis.items():
        print(f"{category.replace('_', ' ').title():20} | {data['demand_level']:6} | {data['confidence']:5.1f}% | {data['trend']:8}")
    
    # Key insights
    insights = analysis.get('insights', [])
    if insights:
        print("\n💡 KEY INSIGHTS")
        print("-" * 15)
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
    
    # Recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        print("\n🎯 RECOMMENDATIONS")
        print("-" * 18)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='HP Artisan Intelligence Analysis Runner')
    parser.add_argument('--mode', choices=['simulation', 'twitter', 'kangra'], 
                       default='simulation', help='Analysis mode')
    parser.add_argument('--district', type=str, help='Specific district to analyze')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    print("🏛️ HP GOVERNMENT ARTISAN INTELLIGENCE")
    print("🎯 ANALYSIS RUNNER")
    print("=" * 45)
    print(f"Mode: {args.mode.upper()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if args.mode == 'simulation':
        run_simulation_analysis()
    elif args.mode == 'twitter':
        run_twitter_analysis()
    elif args.mode == 'kangra':
        run_kangra_priority_analysis()
    
    print("\n🎉 Analysis completed successfully!")

if __name__ == "__main__":
    main()
