import sys
import os
import pytest

# Ensure project root is on sys.path so tests can import local modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from advanced_sentiment_engine import GovernmentGradeArtisanIntelligence


def test_calculate_demand_score_bounds():
    engine = GovernmentGradeArtisanIntelligence(data_mode='simulation')

    # High positive sentiment, large engagement and post count -> high demand
    score = engine.calculate_demand_score(0.9, 1000, 100)
    assert 0.0 <= score <= 1.0
    assert score >= 0.7

    # Negative sentiment and no engagement -> zero demand
    score2 = engine.calculate_demand_score(-0.5, 0, 0)
    assert score2 == 0.0


def test_calculate_trend_cases():
    engine = GovernmentGradeArtisanIntelligence(data_mode='simulation')

    # Rising trend
    assert engine.calculate_trend([0.1, 0.15, 0.2]) == 'RISING'

    # Declining trend
    assert engine.calculate_trend([0.5, 0.3, 0.0]) == 'DECLINING'

    # Insufficient data
    assert engine.calculate_trend([0.1, 0.12]) == 'INSUFFICIENT_DATA'
