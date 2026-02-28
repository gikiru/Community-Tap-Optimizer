import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

class MEALCriteria:
    """Flexible container for M&E and Engineering criteria."""
    def __init__(self, name="Custom Standard", optimal_dist=250, max_dist=400, target_coverage=70, min_hh=20, max_hh=70):
        self.name = name
        self.optimal_dist = optimal_dist
        self.max_dist = max_dist
        self.target_coverage = target_coverage
        self.min_hh = min_hh
        self.max_hh = max_hh

class ImpactEngine:
    """The high-performance core for distance and load analytics."""
    
    @staticmethod
    def assign_taps_vectorized(households_df, taps_df):
        """Ultra-fast spatial indexing using BallTree."""
        hh_coords = np.radians(households_df[['SM Latitude', 'SM Longitude']].values)
        tap_coords = np.radians(taps_df[['SM Latitude', 'SM Longitude']].values)
        
        tree = BallTree(tap_coords, metric='haversine')
        distances, indices = tree.query(hh_coords, k=1)
        
        # Earth radius 6371000m
        households_df['Distance'] = distances.ravel() * 6371000
        households_df['Assigned_Tap'] = taps_df.iloc[indices.ravel()]['SM Title'].values
        return households_df

    @staticmethod
    def calculate_scores(households_df, taps_df, criteria):
        """Generic scoring logic against ANY criteria object."""
        dist = households_df['Distance'].values
        coverage_opt = (dist <= criteria.optimal_dist).sum() / len(dist) * 100
        avg_dist = dist.mean()
        
        # Load balancing
        counts = households_df['Assigned_Tap'].value_counts()
        equity = (counts.std() / counts.mean() * 100) if not counts.empty else 0
        
        return {
            "avg_dist": avg_dist,
            "coverage_optimal": coverage_opt,
            "equity_score": equity,
            "score": (max(0, 100 - (avg_dist/5)) * 0.4 + (100 - equity) * 0.3 + coverage_opt * 0.3)
        }