import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
import io
import os
import base64
from streamlit_folium import st_folium
from geopy.distance import geodesic
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from functools import lru_cache
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
try:
    from scipy.spatial import Voronoi
    from shapely.geometry import Point, Polygon
    VORONOI_AVAILABLE = True
except ImportError:
    VORONOI_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide", page_title="Tap Placement Optimizer v3.0")


# ============================================================================
# CORE DISTANCE CALCULATIONS (Optimized)
# ============================================================================

@st.cache_data
def compute_haversine_distance_matrix(coords1, coords2):
    """Vectorized computation of haversine distances between two sets of coordinates."""
    coords1_rad = np.radians(coords1)
    coords2_rad = np.radians(coords2)
    
    lat1 = coords1_rad[:, 0:1]
    lon1 = coords1_rad[:, 1:2]
    lat2 = coords2_rad[:, 0].reshape(1, -1)
    lon2 = coords2_rad[:, 1].reshape(1, -1)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371000  # Earth radius in meters
    
    return r * c


def assign_nearest_taps(households_df, taps_df, tap_id_col='SM Title'):
    """Ultra-fast assignment using BallTree spatial indexing."""
    if taps_df is None or taps_df.empty:
        return households_df
    
    households_df = households_df.copy()
    
    hh_coords = households_df[['SM Latitude', 'SM Longitude']].values
    tap_coords = taps_df[['SM Latitude', 'SM Longitude']].values
    tap_ids = taps_df[tap_id_col].values
    
    if len(taps_df) > 10:  # Use BallTree for larger datasets
        tap_coords_rad = np.radians(tap_coords)
        tree = BallTree(tap_coords_rad, metric='haversine')
        
        hh_coords_rad = np.radians(hh_coords)
        distances, indices = tree.query(hh_coords_rad, k=1)
        
        distances_meters = distances.ravel() * 6371000
        indices_flat = indices.ravel()
    else:  # Use direct calculation for small datasets
        distances = compute_haversine_distance_matrix(hh_coords, tap_coords)
        indices_flat = distances.argmin(axis=1)
        distances_meters = distances.min(axis=1)
    
    households_df['Nearest Tap'] = tap_ids[indices_flat]
    households_df['Distance to Nearest Tap'] = distances_meters
    
    return households_df


# ============================================================================
# DATA VALIDATION & CLEANING
# ============================================================================

def validate_and_clean_data(df):
    """Comprehensive data validation with automatic cleaning."""
    issues = []
    stats = {}
    
    required_cols = ['SM Latitude', 'SM Longitude']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        issues.append(f"❌ Missing required columns: {missing}")
        return df, issues, stats
    
    stats['original_count'] = len(df)
    
    # Validate coordinate ranges
    invalid_lat = (df['SM Latitude'].abs() > 90) | df['SM Latitude'].isna()
    invalid_lon = (df['SM Longitude'].abs() > 180) | df['SM Longitude'].isna()
    
    if invalid_lat.any():
        issues.append(f"⚠️ {invalid_lat.sum()} invalid latitudes removed")
        df = df[~invalid_lat]
    
    if invalid_lon.any():
        issues.append(f"⚠️ {invalid_lon.sum()} invalid longitudes removed")
        df = df[~invalid_lon]
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['SM Latitude', 'SM Longitude'])
    if duplicates.any():
        issues.append(f"⚠️ {duplicates.sum()} duplicate locations removed")
        df = df.drop_duplicates(subset=['SM Latitude', 'SM Longitude'])
    
    stats['final_count'] = len(df)
    stats['removed_count'] = stats['original_count'] - stats['final_count']
    
    if not issues:
        issues.append("✅ Data validation passed")
    
    return df, issues, stats


# ============================================================================
# SMART PARAMETER RECOMMENDATIONS
# ============================================================================

def recommend_parameters(households_df):
    """Auto-suggest parameters based on dataset characteristics."""
    n_households = len(households_df)
    
    coords = households_df[['SM Latitude', 'SM Longitude']].values
    
    lat_range = coords[:, 0].max() - coords[:, 0].min()
    lon_range = coords[:, 1].max() - coords[:, 1].min()
    
    area_km2 = (lat_range * 111) * (lon_range * 111 * np.cos(np.radians(coords[:, 0].mean())))
    density = n_households / max(area_km2, 0.01)
    
    recommendations = {'reasoning': []}
    
    if density > 100:
        recommendations['k'] = max(3, int(n_households / 80))
        recommendations['min_households'] = 40
        recommendations['max_households'] = 100
        recommendations['reasoning'].append("🏙️ High density (urban)")
    elif density > 30:
        recommendations['k'] = max(3, int(n_households / 60))
        recommendations['min_households'] = 30
        recommendations['max_households'] = 70
        recommendations['reasoning'].append("🏘️ Medium density (suburban)")
    else:
        recommendations['k'] = max(3, int(n_households / 40))
        recommendations['min_households'] = 15
        recommendations['max_households'] = 50
        recommendations['reasoning'].append("🌾 Low density (rural)")
    
    recommendations['reasoning'].append(f"📊 Households: {n_households}")
    recommendations['reasoning'].append(f"📏 Density: {density:.1f} HH/km²")
    recommendations['reasoning'].append(f"📍 Area: {area_km2:.2f} km²")
    
    return recommendations


# ============================================================================
# OPTIMIZATION FUNCTIONS
# ============================================================================

def enforce_max_households(taps_df, households_df, tap_id_col='SM Title', max_households=70):
    """Split taps serving more than max_households."""
    taps = taps_df.copy()
    hhs = households_df.copy()
    
    if taps is None or taps.empty or max_households is None:
        return taps, hhs
    
    max_iterations = 10
    iteration = 0
    
    with st.spinner("Enforcing maximum household constraint..."):
        while iteration < max_iterations:
            iteration += 1
            
            hhs = assign_nearest_taps(hhs, taps, tap_id_col)
            counts = hhs['Nearest Tap'].value_counts()
            
            oversized = counts[counts > max_households].index.tolist()
            
            if not oversized:
                break
            
            new_taps_list = []
            taps_to_remove = []
            
            for tap_id in oversized:
                assigned_hh = hhs[hhs['Nearest Tap'] == tap_id]
                n_assigned = len(assigned_hh)
                
                n_splits = int(np.ceil(n_assigned / max_households))
                
                if n_splits <= 1:
                    continue
                
                coords = assigned_hh[['SM Latitude', 'SM Longitude']].values
                
                try:
                    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10).fit(coords)
                    centroids = kmeans.cluster_centers_
                    
                    for i, centroid in enumerate(centroids):
                        new_tap = {
                            'SM Title': f"{tap_id}_split_{i+1}",
                            'SM Latitude': centroid[0],
                            'SM Longitude': centroid[1]
                        }
                        new_taps_list.append(new_tap)
                    
                    taps_to_remove.append(tap_id)
                except Exception:
                    continue
            
            if new_taps_list:
                taps = taps[~taps[tap_id_col].isin(taps_to_remove)]
                new_taps_df = pd.DataFrame(new_taps_list)
                taps = pd.concat([taps, new_taps_df], ignore_index=True)
            else:
                break
    
    hhs = assign_nearest_taps(hhs, taps, tap_id_col)
    return taps, hhs


def propose_tap_positions(households_df, initial_k, min_households, max_households):
    """Propose optimal tap positions using K-means clustering."""
    
    with st.spinner("Running optimization..."):
        coords = households_df[['SM Latitude', 'SM Longitude']].to_numpy()
        
        k = int(initial_k)
        
        # Run K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(coords)
        centers = kmeans.cluster_centers_
        
        # Create proposed taps
        proposed_taps = pd.DataFrame(centers, columns=['SM Latitude', 'SM Longitude'])
        proposed_taps['SM Title'] = ['Tap_' + str(i + 1) for i in range(len(proposed_taps))]
        
        # Assign households
        households_assign = households_df.copy()
        households_assign = assign_nearest_taps(households_assign, proposed_taps, 'SM Title')
        
        # Enforce max constraint if needed
        if max_households:
            proposed_taps, households_assign = enforce_max_households(
                proposed_taps, households_assign,
                tap_id_col='SM Title',
                max_households=max_households
            )
        
        # Remove under-utilized taps
        if min_households:
            # Check which column to use
            if 'Nearest Tap' in households_assign.columns:
                count_column = 'Nearest Tap'
            elif 'SM Title' in households_assign.columns:
                count_column = 'SM Title'
            else:
                return proposed_taps, households_assign
            
            counts = households_assign[count_column].value_counts()
            valid_taps = counts[counts >= min_households].index
            
            proposed_taps = proposed_taps[proposed_taps['SM Title'].isin(valid_taps)].reset_index(drop=True)
            
            if len(proposed_taps) > 0:
                households_assign = assign_nearest_taps(households_assign, proposed_taps, 'SM Title')
            else:
                st.error("❌ All taps removed by minimum household constraint!")
                st.info(f"💡 Try: Lower min_households (currently {min_households}) or increase number of taps")
        
        return proposed_taps, households_assign


def generate_recommendations(metrics, tap_info):
    """Generate actionable recommendations based on performance metrics."""
    recommendations = {
        'priority': [],  # High priority issues
        'improvements': [],  # Medium priority improvements
        'optimizations': [],  # Nice-to-have optimizations
        'summary': '',
        'rating': '',
        'color': ''
    }
    
    overall_score = metrics['overall_score']
    
    # Determine overall rating
    if overall_score >= 90:
        recommendations['rating'] = 'Excellent ⭐⭐⭐⭐⭐'
        recommendations['color'] = 'success'
        recommendations['summary'] = 'Outstanding tap placement! This configuration provides excellent service coverage with well-balanced loads and minimal walking distances.'
    elif overall_score >= 80:
        recommendations['rating'] = 'Very Good ⭐⭐⭐⭐'
        recommendations['color'] = 'success'
        recommendations['summary'] = 'Very good tap placement with strong performance across key metrics. Minor optimizations could push this to excellent.'
    elif overall_score >= 70:
        recommendations['rating'] = 'Good ⭐⭐⭐'
        recommendations['color'] = 'info'
        recommendations['summary'] = 'Good tap placement that meets basic requirements. Some improvements recommended to enhance service quality.'
    elif overall_score >= 60:
        recommendations['rating'] = 'Fair ⭐⭐'
        recommendations['color'] = 'warning'
        recommendations['summary'] = 'Fair placement with notable issues. Several improvements needed to provide adequate service.'
    else:
        recommendations['rating'] = 'Poor ⭐'
        recommendations['color'] = 'error'
        recommendations['summary'] = 'Significant issues detected. Major redesign recommended to ensure adequate water access.'
    
    # Distance analysis
    avg_dist = metrics['avg_distance']
    coverage_250 = metrics['coverage_250m']
    coverage_400 = metrics['coverage_400m']
    
    if avg_dist > 400:
        recommendations['priority'].append({
            'issue': 'Excessive average walking distance',
            'detail': f'Average distance is {avg_dist:.0f}m (target: <250m)',
            'action': 'Add more taps in underserved areas or relocate existing taps closer to household clusters'
        })
    elif avg_dist > 300:
        recommendations['improvements'].append({
            'issue': 'High average walking distance',
            'detail': f'Average distance is {avg_dist:.0f}m (optimal: <250m)',
            'action': 'Consider adding 1-2 additional taps in areas with longer walking distances'
        })
    
    if coverage_250 < 50:
        recommendations['priority'].append({
            'issue': 'Poor coverage within optimal distance',
            'detail': f'Only {coverage_250:.1f}% of households within 250m',
            'action': 'Increase number of taps significantly - aim for 70%+ coverage within 250m'
        })
    elif coverage_250 < 70:
        recommendations['improvements'].append({
            'issue': 'Moderate coverage within optimal distance',
            'detail': f'{coverage_250:.1f}% of households within 250m (target: 70%+)',
            'action': 'Add taps in areas where households are 250-400m from nearest tap'
        })
    
    if coverage_400 < 80:
        recommendations['priority'].append({
            'issue': 'Households beyond acceptable distance',
            'detail': f'{100-coverage_400:.1f}% of households are >400m from nearest tap',
            'action': 'Urgent: Provide additional taps for households beyond 400m walking distance'
        })
    
    # Load balancing analysis
    equity_score = metrics['equity_score']
    min_hh = int(metrics['min_households'])
    max_hh = int(metrics['max_households'])
    avg_hh = metrics['avg_households_per_tap']
    
    if equity_score > 50:
        recommendations['priority'].append({
            'issue': 'Severe load imbalance',
            'detail': f'Tap loads vary from {min_hh} to {max_hh} households (high variation)',
            'action': 'Redistribute tap locations to balance loads - aim for more uniform household allocation'
        })
    elif equity_score > 30:
        recommendations['improvements'].append({
            'issue': 'Moderate load imbalance',
            'detail': f'Tap loads range from {min_hh} to {max_hh} households',
            'action': 'Adjust tap positions to better balance household distribution'
        })
    
    # Individual tap analysis
    overloaded_taps = tap_info[tap_info['Households'] > avg_hh * 1.5]
    underutilized_taps = tap_info[tap_info['Households'] < avg_hh * 0.5]
    
    if len(overloaded_taps) > 0:
        tap_names = ', '.join(overloaded_taps['Tap Name'].tolist()[:3])
        if len(overloaded_taps) > 3:
            tap_names += f' (+{len(overloaded_taps)-3} more)'
        recommendations['improvements'].append({
            'issue': 'Overloaded taps detected',
            'detail': f'{len(overloaded_taps)} tap(s) serving >50% above average: {tap_names}',
            'action': 'Split overloaded taps or add nearby taps to reduce load'
        })
    
    if len(underutilized_taps) > 0:
        tap_names = ', '.join(underutilized_taps['Tap Name'].tolist()[:3])
        if len(underutilized_taps) > 3:
            tap_names += f' (+{len(underutilized_taps)-3} more)'
        recommendations['improvements'].append({
            'issue': 'Underutilized taps detected',
            'detail': f'{len(underutilized_taps)} tap(s) serving <50% of average: {tap_names}',
            'action': 'Relocate underutilized taps to areas with higher household density'
        })
    
    # Distance hotspots
    far_taps = tap_info[tap_info['Avg Distance (m)'] > 350]
    if len(far_taps) > 0:
        tap_names = ', '.join(far_taps['Tap Name'].tolist()[:3])
        if len(far_taps) > 3:
            tap_names += f' (+{len(far_taps)-3} more)'
        recommendations['improvements'].append({
            'issue': 'Long average distances from some taps',
            'detail': f'{len(far_taps)} tap(s) with avg distance >350m: {tap_names}',
            'action': 'Investigate household distribution around these taps - may need repositioning'
        })
    
    # Optimization opportunities
    if overall_score >= 70 and overall_score < 85:
        recommendations['optimizations'].append({
            'opportunity': 'Fine-tune tap positions',
            'detail': 'Good foundation - small adjustments could improve score',
            'action': 'Review the map for any clusters slightly beyond 250m radius and adjust positions'
        })
    
    if metrics['total_taps'] < 5 and metrics['total_households'] > 300:
        recommendations['optimizations'].append({
            'opportunity': 'Consider additional taps',
            'detail': f'{metrics["total_taps"]} taps for {metrics["total_households"]} households',
            'action': 'Budget permitting, adding 1-2 more taps could significantly improve coverage'
        })
    
    # eWater criteria compliance
    if coverage_250 < 70:
        recommendations['priority'].append({
            'issue': 'Below eWater optimal distance standard',
            'detail': 'eWater criteria recommend >70% coverage within 250m',
            'action': 'Review Community Tap Positioning Criteria Guide - may need additional taps'
        })
    
    if avg_dist > 400:
        recommendations['priority'].append({
            'issue': 'Exceeds eWater acceptable distance threshold',
            'detail': 'eWater criteria set 400m as maximum acceptable average',
            'action': 'Priority action required - this configuration does not meet organizational standards'
        })
    
    return recommendations

# ============================================================================
# DISTANCE BINS FOR DETAILED ANALYSIS
# ============================================================================

DISTANCE_BINS = {
    '<50m': (0, 49),
    '50-99m': (50, 99),
    '100-149m': (100, 149),
    '150-199m': (150, 199),
    '200-249m': (200, 249),
    '250-299m': (250, 299),
    '300-349m': (300, 349),
    '350-399m': (350, 399),
    '400-449m': (400, 449),
    '450-499m': (450, 499),
    '>500m': (500, 999999)
}

# ============================================================================
# STATISTICS & METRICS
# ============================================================================

def compute_tap_stats(households_df, taps_df, tap_id_col='SM Title', bins=None):
    """Compute per-tap statistics with optional distance bins."""
    if bins is None:
        bins = DISTANCE_BINS
    tap_info_list = []
    assigned_groups = households_df.groupby('Nearest Tap')
    
    for _, tap in taps_df.iterrows():
        tap_id = tap.get(tap_id_col, tap.get('SM Title'))
        
        if tap_id in assigned_groups.groups:
            assigned_hhs = assigned_groups.get_group(tap_id)
            distances = assigned_hhs['Distance to Nearest Tap'].values
            
            tap_info = {
                'Tap Name': tap.get('SM Title', ''),
                'Latitude': tap.get('SM Latitude', None),
                'Longitude': tap.get('SM Longitude', None),
                'Households': len(assigned_hhs),
                'Avg Distance (m)': int(distances.mean()) if len(distances) > 0 else 0,
                'Max Distance (m)': int(distances.max()) if len(distances) > 0 else 0,
            }
            # Add distance bins
            for bin_label, (bin_min, bin_max) in bins.items():
                count = ((distances >= bin_min) & (distances <= bin_max)).sum()
                tap_info[bin_label] = int(count)
        else:
            tap_info = {
                'Tap Name': tap.get('SM Title', ''),
                'Latitude': tap.get('SM Latitude', None),
                'Longitude': tap.get('SM Longitude', None),
                'Households': 0,
                'Avg Distance (m)': 0,
                'Max Distance (m)': 0,
            }
            # Zero out all bins
            for bin_label in bins.keys():
                tap_info[bin_label] = 0        
        tap_info_list.append(tap_info)
    
    return pd.DataFrame(tap_info_list)


def compute_performance_metrics(households_df, taps_df):
    """Compute comprehensive performance metrics."""
    metrics = {}
    
    metrics['total_households'] = len(households_df)
    metrics['total_taps'] = len(taps_df)
    metrics['avg_households_per_tap'] = len(households_df) / max(len(taps_df), 1)
    
    distances = households_df['Distance to Nearest Tap'].values
    metrics['avg_distance'] = distances.mean()
    metrics['median_distance'] = np.median(distances)
    metrics['max_distance'] = distances.max()
    
    metrics['coverage_250m'] = (distances <= 250).sum() / len(distances) * 100
    metrics['coverage_400m'] = (distances <= 400).sum() / len(distances) * 100
    metrics['coverage_500m'] = (distances <= 500).sum() / len(distances) * 100
    
    tap_counts = households_df['Nearest Tap'].value_counts()
    metrics['min_households'] = tap_counts.min()
    metrics['max_households'] = tap_counts.max()
    metrics['equity_score'] = tap_counts.std() / max(tap_counts.mean(), 1) * 100
    
    distance_score = max(0, 100 - (metrics['avg_distance'] / 500 * 100))
    equity_score_normalized = max(0, 100 - metrics['equity_score'])
    coverage_score = metrics['coverage_250m']
    
    metrics['overall_score'] = (distance_score * 0.4 + equity_score_normalized * 0.3 + coverage_score * 0.3)
    
    return metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def get_household_color_by_distance(distance):
    """Color code households by distance to nearest tap."""
    if distance < 100:
        return '#00ff00'
    elif distance < 250:
        return '#92D050'
    elif distance < 500:
        return '#FFC000'
    else:
        return '#FF0000'


def create_map(households_df, taps_df, tap_id_col='SM Title', show_coverage=True, 
               coverage_radius=250, show_heatmap=False, color_by_distance=True):
    """Create interactive map."""
    
    map_center = [households_df['SM Latitude'].mean(), households_df['SM Longitude'].mean()]
    m = folium.Map(location=map_center, zoom_start=15)
    
    # Add base layers
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite'
    ).add_to(m)
    
    # Add coverage circles
    if show_coverage and taps_df is not None and not taps_df.empty:
        for _, tap in taps_df.iterrows():
            folium.Circle(
                location=[tap['SM Latitude'], tap['SM Longitude']],
                radius=coverage_radius,
                color='#3388ff',
                weight=2,
                fill=True,
                fillColor='#3388ff',
                fillOpacity=0.1
            ).add_to(m)
    
    # Add heatmap
    if show_heatmap:
        heat_data = households_df[['SM Latitude', 'SM Longitude']].values.tolist()
        HeatMap(heat_data, radius=15, blur=25, max_zoom=13, name='Density Heatmap').add_to(m)
    
    # Add households
    if color_by_distance and 'Distance to Nearest Tap' in households_df.columns:
        for _, row in households_df.iterrows():
            color = get_household_color_by_distance(row['Distance to Nearest Tap'])
            folium.CircleMarker(
                location=(row['SM Latitude'], row['SM Longitude']),
                radius=3,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"Distance: {row['Distance to Nearest Tap']:.0f}m"
            ).add_to(m)
    else:
        for _, row in households_df.iterrows():
            folium.CircleMarker(
                location=(row['SM Latitude'], row['SM Longitude']),
                radius=3,
                color='#3388ff',
                fill=True
            ).add_to(m)
    
    # Add tap markers
    if taps_df is not None and not taps_df.empty:
        tap_counts = households_df.groupby('Nearest Tap').size().to_dict()
        
        for _, tap in taps_df.iterrows():
            tap_id = tap.get(tap_id_col, '')
            count = tap_counts.get(tap_id, 0)
            
            # Color based on load
            if count < 20:
                color = 'gray'
            elif count < 50:
                color = 'orange'
            elif count < 70:
                color = 'green'
            else:
                color = 'red'
            
            folium.Marker(
                location=[tap['SM Latitude'], tap['SM Longitude']],
                popup=f"<b>{tap_id}</b><br>{count} households",
                icon=folium.Icon(color=color, icon='tint', prefix='fa')
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; width: 200px; 
                background-color: white; border: 2px solid #333; z-index: 9999; 
                font-size: 13px; padding: 12px; border-radius: 5px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
    <p style="margin: 0 0 8px 0; font-weight: bold; color: #333; font-size: 14px;">Distance Legend</p>
    <p style="margin: 4px 0; color: #333;">
        <span style="color: #00ff00; font-size: 16px;">●</span> 
        <span style="color: #333;">&lt; 100m (Excellent)</span>
    </p>
    <p style="margin: 4px 0; color: #333;">
        <span style="color: #92D050; font-size: 16px;">●</span> 
        <span style="color: #333;">100-250m (Good)</span>
    </p>
    <p style="margin: 4px 0; color: #333;">
        <span style="color: #FFC000; font-size: 16px;">●</span> 
        <span style="color: #333;">250-500m (Acceptable)</span>
    </p>
    <p style="margin: 4px 0; color: #333;">
        <span style="color: #FF0000; font-size: 16px;">●</span> 
        <span style="color: #333;">&gt; 500m (Poor)</span>
    </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    folium.LayerControl().add_to(m)
    
    return m


# ============================================================================
# CHARTS
# ============================================================================

def create_distance_chart(households_df):
    """Create distance distribution chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    distances = households_df['Distance to Nearest Tap'].values
    ax.hist(distances, bins=30, color='#3388ff', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Distance (meters)', fontsize=12)
    ax.set_ylabel('Number of Households', fontsize=12)
    ax.set_title('Distance Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_tap_load_chart(tap_info_df):
    """Create tap load bar chart with average distance overlay."""
    # Safety check
    if tap_info_df is None or tap_info_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No tap data available', ha='center', va='center')
        return fig
    
    # Check for required columns
    required_cols = ['Tap Name', 'Households', 'Avg Distance (m)']
    missing_cols = [col for col in required_cols if col not in tap_info_df.columns]
    
    if missing_cols:
        fig, ax = plt.subplots(figsize=(12, 6))
        msg = f"Missing columns: {missing_cols}\nAvailable: {list(tap_info_df.columns)}"
        ax.text(0.5, 0.5, msg, ha='center', va='center')
        return fig
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # X-axis positions
    x_pos = np.arange(len(tap_info_df))
    
    # Plot 1: Households (bars) - LEFT Y-AXIS
    bars = ax1.bar(x_pos, tap_info_df['Households'], 
                   color='#3388ff', edgecolor='black', alpha=0.7, 
                   width=0.6, label='Households')
    
    # Color code bars by load
    for i, count in enumerate(tap_info_df['Households']):
        if count < 20:
            bars[i].set_color('#ff6b6b')  # Red - underutilized
        elif count < 50:
            bars[i].set_color('#feca57')  # Yellow - good
        elif count < 70:
            bars[i].set_color('#1dd1a1')  # Green - optimal
        else:
            bars[i].set_color('#ff6348')  # Dark red - overloaded
    
    # Configure left y-axis (Households)
    ax1.set_xlabel('Tap Name', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Households Served', fontsize=12, fontweight='bold', color='#3388ff')
    ax1.tick_params(axis='y', labelcolor='#3388ff')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(tap_info_df['Tap Name'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Add reference lines for ideal household range
    ax1.axhline(y=20, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axhline(y=70, color='green', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(len(tap_info_df) - 0.5, 20, 'Min (20)', fontsize=9, color='red', 
             verticalalignment='bottom', horizontalalignment='right')
    ax1.text(len(tap_info_df) - 0.5, 70, 'Max (70)', fontsize=9, color='green',
             verticalalignment='bottom', horizontalalignment='right')
    
    # Plot 2: Average Distance (line) - RIGHT Y-AXIS
    ax2 = ax1.twinx()
    line = ax2.plot(x_pos, tap_info_df['Avg Distance (m)'], 
                    color='#FF6B35', marker='o', linewidth=2.5, 
                    markersize=8, label='Avg Distance',
                    markeredgecolor='white', markeredgewidth=1.5)
    
    # Configure right y-axis (Distance)
    ax2.set_ylabel('Average Distance (m)', fontsize=12, fontweight='bold', color='#FF6B35')
    ax2.tick_params(axis='y', labelcolor='#FF6B35')
    
    # Add reference line for optimal distance
    ax2.axhline(y=250, color='orange', linestyle='--', alpha=0.5, linewidth=1)
    ax2.text(0, 250, '250m (optimal)', fontsize=9, color='orange',
             verticalalignment='bottom', horizontalalignment='left')
    
    # Title and legend
    ax1.set_title('Tap Load Distribution & Average Distance', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Create combined legend
    bars_legend = plt.Rectangle((0, 0), 1, 1, fc='#3388ff', alpha=0.7, edgecolor='black')
    lines_legend = plt.Line2D([0], [0], color='#FF6B35', marker='o', linewidth=2.5, markersize=8)
    ax1.legend([bars_legend, lines_legend], ['Households', 'Avg Distance'], 
               loc='upper left', framealpha=0.9)
    
    plt.tight_layout()
    return fig

def create_distance_chart_plotly(households_df):
    """Create interactive distance distribution chart using Plotly."""
    if 'Distance to Nearest Tap' not in households_df.columns:
        return None
    
    distances = households_df['Distance to Nearest Tap']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=distances,
        nbinsx=30,
        marker_color='#3388ff',
        marker_line_color='white',
        marker_line_width=1,
        opacity=0.8,
        name='Households'
    ))
    
    fig.update_layout(
        title='Distance Distribution',
        xaxis_title='Distance to Nearest Tap (m)',
        yaxis_title='Number of Households',
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    # Add reference lines
    fig.add_vline(x=250, line_dash="dash", line_color="green", 
                  annotation_text="250m (optimal)")
    fig.add_vline(x=400, line_dash="dash", line_color="orange",
                  annotation_text="400m (max acceptable)")
    
    return fig


def create_tap_load_chart_plotly(tap_info_df):
    """Create interactive tap load chart with average distance using Plotly."""
    if tap_info_df is None or tap_info_df.empty or 'Households' not in tap_info_df.columns:
        return None
    
    # Check for Avg Distance column
    if 'Avg Distance (m)' not in tap_info_df.columns:
        # Fall back to original version
        colors = []
        for count in tap_info_df['Households']:
            if count < 20:
                colors.append('#ff6b6b')
            elif count < 50:
                colors.append('#feca57')
            elif count < 70:
                colors.append('#1dd1a1')
            else:
                colors.append('#ff6348')
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=tap_info_df['Tap Name'],
            y=tap_info_df['Households'],
            marker_color=colors,
            marker_line_color='rgba(0,0,0,0.3)',
            marker_line_width=1.5,
            text=tap_info_df['Households'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Households: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Tap Load Distribution',
            xaxis_title='Tap Name',
            yaxis_title='Households Served',
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        fig.add_hline(y=20, line_dash="dash", line_color="red",
                      annotation_text="Min (20)")
        fig.add_hline(y=70, line_dash="dash", line_color="green",
                      annotation_text="Max (70)")
        
        return fig
    
    # Color code by load
    colors = []
    for count in tap_info_df['Households']:
        if count < 20:
            colors.append('#ff6b6b')
        elif count < 50:
            colors.append('#feca57')
        elif count < 70:
            colors.append('#1dd1a1')
        else:
            colors.append('#ff6348')
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add bar chart for households
    fig.add_trace(go.Bar(
        x=tap_info_df['Tap Name'],
        y=tap_info_df['Households'],
        name='Households',
        marker_color=colors,
        marker_line_color='rgba(0,0,0,0.3)',
        marker_line_width=1.5,
        text=tap_info_df['Households'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Households: %{y}<br><extra></extra>',
        yaxis='y'
    ))
    
    # Add line chart for average distance
    fig.add_trace(go.Scatter(
        x=tap_info_df['Tap Name'],
        y=tap_info_df['Avg Distance (m)'],
        name='Avg Distance',
        mode='lines+markers',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=10, color='#FF6B35', line=dict(color='white', width=2)),
        hovertemplate='<b>%{x}</b><br>Avg Distance: %{y:.0f}m<br><extra></extra>',
        yaxis='y2'
    ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title='Tap Load Distribution & Average Distance',
        xaxis=dict(title="Tap Name", tickangle=-45),
        yaxis=dict(
            title=dict(text='<b>Households Served</b>', font=dict(color='#3388ff')),
            tickfont=dict(color='#3388ff'),
            side='left'
        ),
        yaxis2=dict(
            title=dict(text='<b>Average Distance (m)</b>', font=dict(color='#FF6B35')),
            tickfont=dict(color='#FF6B35'),
            overlaying='y',
            side='right'
        ),
        template='plotly_white',
        height=450,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    # Add reference lines
    fig.add_hline(y=20, line_dash="dash", line_color="red",
                  annotation_text="Min (20)", annotation_position="right", yref="y")
    fig.add_hline(y=70, line_dash="dash", line_color="green",
                  annotation_text="Max (70)", annotation_position="right", yref="y")
    fig.add_hline(y=250, line_dash="dash", line_color="orange",
                  annotation_text="250m (optimal)", annotation_position="left", yref="y2")
    
    return fig

def create_coverage_pie_chart(households_df):
    """Create coverage pie chart by distance bins."""
    if 'Distance to Nearest Tap' not in households_df.columns:
        return None
    
    distances = households_df['Distance to Nearest Tap']
    
    bins_data = []
    for label, (min_dist, max_dist) in DISTANCE_BINS.items():
        count = ((distances >= min_dist) & (distances <= max_dist)).sum()
        bins_data.append({'Range': label, 'Count': count})
    
    df = pd.DataFrame(bins_data)
    
    colors = ['#00ff00', '#92D050', '#C5E0B4', '#FFC000', '#F4B084', 
              '#FF6B6B', '#E74C3C', '#C0392B', '#7F0000']
    
    fig = go.Figure(data=[go.Pie(
        labels=df['Range'],
        values=df['Count'],
        marker=dict(colors=colors),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Households: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Coverage Distribution by Distance',
        template='plotly_white',
        height=400
    )
    
    return fig

def add_voronoi_boundaries(m, taps_df):
    """Add Voronoi diagram to show service area boundaries."""
    if not VORONOI_AVAILABLE:
        return
    
    if len(taps_df) < 4:
        return  # Need at least 4 points
    
    try:
        points = taps_df[['SM Latitude', 'SM Longitude']].values
        vor = Voronoi(points)
        
        # Get bounds for clipping
        lats = taps_df['SM Latitude']
        lons = taps_df['SM Longitude']
        lat_margin = (lats.max() - lats.min()) * 0.2
        lon_margin = (lons.max() - lons.min()) * 0.2
        
        bounds = [
            [lats.min() - lat_margin, lons.min() - lon_margin],
            [lats.max() + lat_margin, lons.max() + lon_margin]
        ]
        
        # Draw Voronoi regions
        for region_idx in vor.point_region:
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                polygon_points = [vor.vertices[i] for i in region]
                
                # Clip to bounds
                clipped = []
                for lat, lon in polygon_points:
                    lat = max(bounds[0][0], min(bounds[1][0], lat))
                    lon = max(bounds[0][1], min(bounds[1][1], lon))
                    clipped.append([lat, lon])
                
                folium.Polygon(
                    locations=clipped,
                    color='#3388ff',
                    weight=2,
                    fill=False,
                    opacity=0.6,
                    popup='Service Area Boundary'
                ).add_to(m)
    except Exception as e:
        st.warning(f"Could not draw Voronoi boundaries: {e}")

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Title
    st.title("🚰 Tap Placement Optimizer")
    st.markdown("**Streamlined Workflow:** Upload → Optimize → Download → Field-Adjust → Re-upload → Evaluate")
    
    # Quick guide
    with st.expander("📖 How to Use", expanded=False):
        st.markdown("""
        ### Workflow
        1. **Upload Households CSV** - Your household locations
        2. **Get Recommendations** - App suggests optimal parameters
        3. **Propose Taps** - Generate optimized tap locations
        4. **Download** - Take proposed taps to the field
        5. **Adjust in Field** - Make on-site modifications to tap locations
        6. **Upload Adjusted Taps** - Re-upload your field-verified taps
        7. **Auto-Evaluate** - System automatically shows performance
        
        ### Two Use Cases
        **Case 1: New Installation**
        - Upload households → Propose taps → Download for field work
        
        **Case 2: Verify Existing Taps' Performance**  
        - Upload households → Upload existing taps → See performance immediately
        """)
    
    st.markdown("---")
    
    # Initialize session state
    if 'households_df' not in st.session_state:
        st.session_state['households_df'] = None
    if 'taps_df' not in st.session_state:
        st.session_state['taps_df'] = None
    if 'proposed_taps' not in st.session_state:
        st.session_state['proposed_taps'] = None
    if 'proposed_assignments' not in st.session_state:
        st.session_state['proposed_assignments'] = None
    if 'proposed_metrics' not in st.session_state:
        st.session_state['proposed_metrics'] = None
    if 'proposed_tap_info' not in st.session_state:
        st.session_state['proposed_tap_info'] = None
    if 'evaluated_households' not in st.session_state:
        st.session_state['evaluated_households'] = None
    if 'evaluated_metrics' not in st.session_state:
        st.session_state['evaluated_metrics'] = None
    if 'evaluated_tap_info' not in st.session_state:
        st.session_state['evaluated_tap_info'] = None
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        
        hh_file = st.file_uploader("Upload Households CSV", type=['csv'], key='hh_upload')
        
        # Handle CSV upload with proper encoding
        if hh_file is not None:
            try:
                # Read file bytes first, then decode
                bytes_data = hh_file.read()
                hh_file.seek(0)  # Reset file pointer
                
                # Try reading with different methods
                try:
                    households_df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8-sig')
                except:
                    try:
                        households_df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1')
                    except:
                        households_df = pd.read_csv(io.StringIO(bytes_data.decode('utf-8', errors='ignore')))
                
                households_df, issues, stats = validate_and_clean_data(households_df)
                
                with st.expander("📊 Validation Report", expanded=False):
                    for issue in issues:
                        if "✅" in issue:
                            st.success(issue)
                        elif "❌" in issue:
                            st.error(issue)
                        else:
                            st.warning(issue)
                    
                    if stats:
                        st.write(f"**Loaded:** {stats.get('final_count', 0)} households")
                
                # Store in session
                st.session_state['households_df'] = households_df
                
                # Smart recommendations
                if st.button("🔮 Get Smart Recommendations"):
                    recommendations = recommend_parameters(households_df)
                    
                    st.success("**Recommended Settings:**")
                    for reason in recommendations['reasoning']:
                        st.write(reason)
                    
                    st.session_state['rec_k'] = recommendations['k']
                    st.session_state['rec_min'] = recommendations['min_households']
                    st.session_state['rec_max'] = recommendations['max_households']
                    
                    st.info("💡 Click 'Apply Recommendations' below")
            
            except Exception as e:
                st.error(f"Error reading Households CSV: {e}")
                st.info("Try using the cleaned version or check file format")
        
        st.markdown("---")
        st.header("⚙️ Optimization Settings")
        
        # Apply recommendations button
        if st.button("✅ Apply Recommendations"):
            if 'rec_k' in st.session_state:
                st.session_state['param_k'] = st.session_state['rec_k']
                st.session_state['param_min'] = st.session_state['rec_min']
                st.session_state['param_max'] = st.session_state['rec_max']
                st.success("✅ Recommendations applied! Adjust any slider to see the new values.")
                st.rerun()
        
        num_taps = st.slider(
            "Number of taps",
            min_value=1,
            max_value=50,
            value=st.session_state.get('param_k', 5),
            help="How many water taps to propose"
        )
        
        min_households = st.slider(
            "Min households per tap",
            min_value=5,
            max_value=100,
            value=st.session_state.get('param_min', 20),
            step=5,
            help="Taps serving fewer will be removed"
        )
        
        max_households = st.slider(
            "Max households per tap",
            min_value=20,
            max_value=200,
            value=st.session_state.get('param_max', 70),
            step=5,
            help="Taps serving more will be split"
        )
        
        if max_households <= min_households:
            st.error(f"⚠️ Max must be > Min")
        
        # Optional: Upload existing taps for evaluation
        st.markdown("---")
        st.header("📍 Existing Taps (Optional)")
        st.caption("Upload field-verified tap locations to evaluate performance")
        
        taps_file = st.file_uploader("Upload existing/adjusted taps CSV", type=['csv'], key='taps_upload')
        
        if taps_file is not None:
            try:
                bytes_data = taps_file.read()
                taps_file.seek(0)
                
                try:
                    taps_df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8-sig')
                except:
                    taps_df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1')
                
                taps_df, _, _ = validate_and_clean_data(taps_df)
                
                if 'SM Title' not in taps_df.columns:
                    taps_df['SM Title'] = ['Tap_' + str(i+1) for i in range(len(taps_df))]
                
                st.session_state['taps_df'] = taps_df
                st.success(f"✅ Loaded {len(taps_df)} taps")
                
                # Auto-evaluate when taps are uploaded
                if st.session_state.get('households_df') is not None:
                    with st.spinner("Evaluating tap performance..."):
                        households_with_taps = assign_nearest_taps(
                            st.session_state['households_df'], 
                            taps_df, 
                            'SM Title'
                        )
                        tap_info = compute_tap_stats(households_with_taps, taps_df, 'SM Title')
                        metrics = compute_performance_metrics(households_with_taps, taps_df)
                        
                        st.session_state['evaluated_households'] = households_with_taps
                        st.session_state['evaluated_metrics'] = metrics
                        st.session_state['evaluated_tap_info'] = tap_info
                    
                    st.success("✅ Evaluation complete - see results below")
            
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("---")
        st.header("🎨 Visualization Options")
        
        # Chart type
        use_plotly = st.checkbox("Use interactive charts (Plotly)", value=True,
                                help="Interactive charts with zoom and hover. Uncheck for static matplotlib.")
        
        # Map options
        st.subheader("Map Display")
        show_coverage = st.checkbox("Show coverage circles", value=True)
        if show_coverage:
            coverage_radius = st.slider("Coverage radius (m)", 50, 500, 250, 50)
        else:
            coverage_radius = 250
        
        show_heatmap = st.checkbox("Show density heatmap", value=False)
        show_voronoi = st.checkbox("Show service boundaries (Voronoi)", value=False,
                                   disabled=not VORONOI_AVAILABLE,
                                   help="Requires scipy and shapely packages")
        color_by_distance = st.checkbox("Color households by distance", value=True)
        
        st.markdown("---")
        st.header("🎬 Actions")
        
        propose_btn = st.button("🚀 Propose New Taps", use_container_width=True,
                               disabled=st.session_state.get('households_df') is None,
                               help="Generate optimized tap locations based on parameters above")
    
    # Main content area
    households_df = st.session_state.get('households_df')
    taps_df = st.session_state.get('taps_df')
    
    if households_df is None:
        st.info("👈 Upload a Households CSV file to begin")
        st.markdown("### Required CSV Format")
        st.code("""SM Latitude,SM Longitude
-1.286389,36.817223
-1.292156,36.821889
-1.298745,36.815234""")
        return
    
    # Propose new taps
    if propose_btn:
        with st.spinner("Optimizing tap positions..."):
            proposed_taps, proposed_assignments = propose_tap_positions(
                households_df,
                initial_k=num_taps,
                min_households=min_households,
                max_households=max_households
            )
            
            # Compute stats
            tap_info = compute_tap_stats(proposed_assignments, proposed_taps, 'SM Title')
            metrics = compute_performance_metrics(proposed_assignments, proposed_taps)
            
            # Store results
            st.session_state['proposed_taps'] = proposed_taps
            st.session_state['proposed_assignments'] = proposed_assignments
            st.session_state['proposed_metrics'] = metrics
            st.session_state['proposed_tap_info'] = tap_info
        
        counts = proposed_assignments['Nearest Tap'].value_counts()
        st.success(f"✅ Proposed {len(proposed_taps)} taps | Min: {counts.min()} | Max: {counts.max()} | Avg: {counts.mean():.1f} HH/tap")
    
    
    # Display results - EVALUATED taps
    if ('evaluated_households' in st.session_state and 'evaluated_metrics' in st.session_state and
        st.session_state['evaluated_metrics'] is not None):
        st.header("📊 Existing Taps Performance")
        
        metrics = st.session_state['evaluated_metrics']
        tap_info = st.session_state['evaluated_tap_info']
        households_display = st.session_state['evaluated_households']
        taps_display = taps_df
        
        # Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Taps", metrics['total_taps'])
            st.metric("Avg Distance", f"{metrics['avg_distance']:.0f}m")
        with col2:
            st.metric("Coverage (250m)", f"{metrics['coverage_250m']:.1f}%")
            st.metric("Coverage (400m)", f"{metrics['coverage_400m']:.1f}%")
        with col3:
            st.metric("Min HH/Tap", int(metrics['min_households']))
            st.metric("Max HH/Tap", int(metrics['max_households']))
        with col4:
            st.metric("Avg HH/Tap", f"{metrics['avg_households_per_tap']:.1f}")
            st.metric("Equity Score", f"{metrics['equity_score']:.1f}")
        with col5:
            st.metric("Overall Score", f"{metrics['overall_score']:.1f}/100")
            
            # Add visual indicator
            if metrics['overall_score'] >= 90:
                st.success("⭐⭐⭐⭐⭐")
            elif metrics['overall_score'] >= 80:
                st.success("⭐⭐⭐⭐")
            elif metrics['overall_score'] >= 70:
                st.info("⭐⭐⭐")
            elif metrics['overall_score'] >= 60:
                st.warning("⭐⭐")
            else:
                st.error("⭐")

       # Equity Score Explanation
        with st.expander("ℹ️ Understanding Equity Score", expanded=False):
            st.markdown("""
            **Equity Score** measures how evenly households are distributed across taps.
            
            **Score Interpretation:**
            - **0-20:** ✅ Excellent - Very balanced distribution
            - **20-30:** ✅ Good - Acceptable variation
            - **30-50:** ⚠️ Fair - Some imbalance present
            - **>50:** ❌ Poor - Severe load imbalance
            
            **Lower is better!** A score of 0 means perfect balance.
            
            **Why it matters:**
            - Balanced loads = No overloaded taps
            - Even distribution = Fair service for all
            - Low imbalance = No long queues at some taps
            
            **Current Status:**
            """)
            
            if metrics['equity_score'] < 20:
                st.success("✅ Excellent balance - taps have similar loads")
            elif metrics['equity_score'] < 30:
                st.success("✅ Good balance - minor variations acceptable")
            elif metrics['equity_score'] < 50:
                st.warning("⚠️ Fair balance - consider adjusting tap positions")
            else:
                st.error("❌ Poor balance - major imbalance detected")
                st.info(f"💡 Tip: Current range is {int(metrics['min_households'])}-{int(metrics['max_households'])} households per tap. Try adjusting tap positions or increasing number of taps.")

        # Overall Score Explanation
        with st.expander("ℹ️ Understanding Overall Score", expanded=False):
            st.markdown("""
            ### Overall Score Calculation
            
            The **Overall Score (0-100)** is calculated using three weighted components:
            - **Distance Score (40%)**: Rewards lower average walking distances to taps.
            - **Equity Score (30%)**: Rewards more balanced household distribution across taps.
            - **Coverage Score (30%)**: Rewards higher % of households within 250m of a tap.
            
            **Formula:**  
            `Overall Score = 0.4 × Distance Score + 0.3 × Equity Score + 0.3 × Coverage Score`
            
            - **Distance Score:** 100 if avg distance is 0m, 0 if 500m or more.
            - **Equity Score:** 100 if perfect balance, 0 if extreme imbalance.
            - **Coverage Score:** % of households within 250m.
            
            **Aim for a score above 80 for best performance!**
            """)

        # Charts
        st.subheader("📈 Analytics")
        
        if use_plotly:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_distance_chart_plotly(households_display)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = create_tap_load_chart_plotly(tap_info)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Pie chart
            fig3 = create_coverage_pie_chart(households_display)
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_distance_chart(households_display)
                st.pyplot(fig1)
                plt.close(fig1)
            with col2:
                fig2 = create_tap_load_chart(tap_info)
                st.pyplot(fig2)
                plt.close(fig2)
        
        # Recommendations Report
        st.subheader("💡 Performance Analysis & Recommendations")
        
        recommendations = generate_recommendations(metrics, tap_info)
        
        # Overall rating
        if recommendations['color'] == 'success':
            st.success(f"**{recommendations['rating']}** - {recommendations['summary']}")
        elif recommendations['color'] == 'info':
            st.info(f"**{recommendations['rating']}** - {recommendations['summary']}")
        elif recommendations['color'] == 'warning':
            st.warning(f"**{recommendations['rating']}** - {recommendations['summary']}")
        else:
            st.error(f"**{recommendations['rating']}** - {recommendations['summary']}")
        
        # Priority issues
        if recommendations['priority']:
            st.markdown("### 🔴 Priority Issues")
            for i, item in enumerate(recommendations['priority'], 1):
                with st.expander(f"{i}. {item['issue']}", expanded=True):
                    st.write(f"**Issue:** {item['detail']}")
                    st.write(f"**Action:** {item['action']}")
        
        # Improvements
        if recommendations['improvements']:
            st.markdown("### 🟡 Recommended Improvements")
            for i, item in enumerate(recommendations['improvements'], 1):
                with st.expander(f"{i}. {item['issue']}"):
                    st.write(f"**Details:** {item['detail']}")
                    st.write(f"**Action:** {item['action']}")
        
        # Optimizations
        if recommendations['optimizations']:
            st.markdown("### 🟢 Optimization Opportunities")
            for i, item in enumerate(recommendations['optimizations'], 1):
                st.write(f"**{i}. {item['opportunity']}** - {item['detail']}")
                st.caption(f"💡 {item['action']}")
        
        # If perfect score
        if not recommendations['priority'] and not recommendations['improvements']:
            st.success("🎉 No major issues detected! This is an excellent tap configuration.")
        
        # Map
        m = create_map(households_display, taps_display, 'SM Title',
                      show_coverage=show_coverage,
                      coverage_radius=coverage_radius,
                      show_heatmap=show_heatmap,
                      color_by_distance=color_by_distance)
        
        # Add Voronoi if requested
        if show_voronoi:
            add_voronoi_boundaries(m, taps_display)
        
        st_folium(m, width=1200, height=600)
        
        # Downloads
        st.subheader("⬇️ Downloads")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = tap_info.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Tap Statistics", data=csv, file_name="tap_stats.csv", mime='text/csv')
        
        with col2:
            html_str = m.get_root().render()
            st.download_button("🗺️ Map HTML", data=html_str.encode('utf-8'), file_name="map.html", mime='text/html')
        
        with col3:
            csv2 = households_display.to_csv(index=False).encode('utf-8')
            st.download_button("📋 Assignments", data=csv2, file_name="assignments.csv", mime='text/csv')
        
        # Details
        with st.expander("📊 Detailed Statistics"):
            st.dataframe(tap_info, use_container_width=True)
    
    # Display results - PROPOSED taps
    if ('proposed_taps' in st.session_state and 'proposed_metrics' in st.session_state and
        st.session_state['proposed_metrics'] is not None):
        st.header("🎯 Proposed Tap Configuration")
        
        metrics = st.session_state['proposed_metrics']
        tap_info = st.session_state['proposed_tap_info']
        households_display = st.session_state['proposed_assignments']
        taps_display = st.session_state['proposed_taps']
        
        # Metrics Display
        st.subheader("📊 Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Taps", metrics['total_taps'])
            st.metric("Avg Distance", f"{metrics['avg_distance']:.0f}m")
        
        with col2:
            st.metric("Coverage (250m)", f"{metrics['coverage_250m']:.1f}%")
            st.metric("Coverage (400m)", f"{metrics['coverage_400m']:.1f}%")
        
        with col3:
            st.metric("Min HH/Tap", int(metrics['min_households']))
            st.metric("Max HH/Tap", int(metrics['max_households']))
        
        with col4:
            st.metric("Avg HH/Tap", f"{metrics['avg_households_per_tap']:.1f}")
            st.metric("Equity Score", f"{metrics['equity_score']:.1f}")
        
        with col5:
            st.metric("Overall Score", f"{metrics['overall_score']:.1f}/100")
            
            # Visual indicator
            if metrics['overall_score'] >= 90:
                st.success("⭐⭐⭐⭐⭐")
            elif metrics['overall_score'] >= 80:
                st.success("⭐⭐⭐⭐")
            elif metrics['overall_score'] >= 70:
                st.info("⭐⭐⭐")
            elif metrics['overall_score'] >= 60:
                st.warning("⭐⭐")
            else:
                st.error("⭐")
        
        # Equity Score Explanation
        with st.expander("ℹ️ Understanding Equity Score", expanded=False):
            st.markdown("""
            **Equity Score** measures how evenly households are distributed across taps.
            
            **Score Interpretation:**
            - **0-20:** ✅ Excellent - Very balanced distribution
            - **20-30:** ✅ Good - Acceptable variation
            - **30-50:** ⚠️ Fair - Some imbalance present
            - **>50:** ❌ Poor - Severe load imbalance
            
            **Lower is better!** A score of 0 means perfect balance.
            
            **Why it matters:**
            - Balanced loads = No overloaded taps
            - Even distribution = Fair service for all
            - Low imbalance = No long queues at some taps
            
            **Current Status:**
            """)
            
            if metrics['equity_score'] < 20:
                st.success("✅ Excellent balance - taps have similar loads")
            elif metrics['equity_score'] < 30:
                st.success("✅ Good balance - minor variations acceptable")
            elif metrics['equity_score'] < 50:
                st.warning("⚠️ Fair balance - consider adjusting tap positions")
            else:
                st.error("❌ Poor balance - major imbalance detected")
                st.info(f"💡 Tip: Current range is {int(metrics['min_households'])}-{int(metrics['max_households'])} households per tap. Try adjusting tap positions or increasing number of taps.")
        
        # Overall Score Explanation
        with st.expander("ℹ️ Understanding Overall Score", expanded=False):
            st.markdown("""
            ### Overall Score Calculation
            
            The **Overall Score (0-100)** is calculated using three weighted components:
            - **Distance Score (40%)**: Rewards lower average walking distances to taps.
            - **Equity Score (30%)**: Rewards more balanced household distribution across taps.
            - **Coverage Score (30%)**: Rewards higher % of households within 250m of a tap.
            
            **Formula:**  
            `Overall Score = 0.4 × Distance Score + 0.3 × Equity Score + 0.3 × Coverage Score`
            
            - **Distance Score:** 100 if avg distance is 0m, 0 if 500m or more.
            - **Equity Score:** 100 if perfect balance, 0 if extreme imbalance.
            - **Coverage Score:** % of households within 250m.
            
            **Aim for a score above 80 for best performance!**
            """)
        
        # Charts
        st.subheader("📈 Analytics")
        
        if use_plotly:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_distance_chart_plotly(households_display)
                if fig1:
                    st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = create_tap_load_chart_plotly(tap_info)
                if fig2:
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Pie chart
            fig3 = create_coverage_pie_chart(households_display)
            if fig3:
                st.plotly_chart(fig3, use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            with col1:
                fig1 = create_distance_chart(households_display)
                st.pyplot(fig1)
                plt.close(fig1)
            with col2:
                fig2 = create_tap_load_chart(tap_info)
                st.pyplot(fig2)
                plt.close(fig2)
        
        # Recommendations Report
        st.subheader("💡 Performance Analysis & Recommendations")
        
        recommendations = generate_recommendations(metrics, tap_info)
        
        # Overall rating
        if recommendations['color'] == 'success':
            st.success(f"**{recommendations['rating']}** - {recommendations['summary']}")
        elif recommendations['color'] == 'info':
            st.info(f"**{recommendations['rating']}** - {recommendations['summary']}")
        elif recommendations['color'] == 'warning':
            st.warning(f"**{recommendations['rating']}** - {recommendations['summary']}")
        else:
            st.error(f"**{recommendations['rating']}** - {recommendations['summary']}")
        
        # Priority issues
        if recommendations['priority']:
            st.markdown("### 🔴 Priority Issues")
            for i, item in enumerate(recommendations['priority'], 1):
                with st.expander(f"{i}. {item['issue']}", expanded=True):
                    st.write(f"**Issue:** {item['detail']}")
                    st.write(f"**Action:** {item['action']}")
        
        # Improvements
        if recommendations['improvements']:
            st.markdown("### 🟡 Recommended Improvements")
            for i, item in enumerate(recommendations['improvements'], 1):
                with st.expander(f"{i}. {item['issue']}"):
                    st.write(f"**Details:** {item['detail']}")
                    st.write(f"**Action:** {item['action']}")
        
        # Optimizations
        if recommendations['optimizations']:
            st.markdown("### 🟢 Optimization Opportunities")
            for i, item in enumerate(recommendations['optimizations'], 1):
                st.write(f"**{i}. {item['opportunity']}** - {item['detail']}")
                st.caption(f"💡 {item['action']}")
        
        # If perfect score
        if not recommendations['priority'] and not recommendations['improvements']:
            st.success("🎉 No major issues detected! This is an excellent tap configuration.")
        
        # Map
        st.subheader("🗺️ Map View")
        m = create_map(households_display, taps_display, 'SM Title',
                      show_coverage=show_coverage,
                      coverage_radius=coverage_radius,
                      show_heatmap=show_heatmap,
                      color_by_distance=color_by_distance)
        
        # Add Voronoi if requested
        if show_voronoi:
            add_voronoi_boundaries(m, taps_display)
        
        st_folium(m, width=1200, height=600)
        
        # Downloads
        st.subheader("⬇️ Downloads")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tap_info_proposed = compute_tap_stats(households_display, taps_display, 'SM Title')
            csv = tap_info_proposed.to_csv(index=False).encode('utf-8')
            st.download_button("📄 Tap Statistics", data=csv, 
                              file_name="proposed_tap_stats.csv", mime='text/csv')
        
        with col2:
            csv2 = taps_display.to_csv(index=False).encode('utf-8')
            st.download_button("📍 Tap Locations", data=csv2, 
                              file_name="proposed_taps.csv", mime='text/csv')
        
        with col3:
            csv3 = households_display.to_csv(index=False).encode('utf-8')
            st.download_button("📋 Assignments", data=csv3, 
                              file_name="proposed_assignments.csv", mime='text/csv')        
        with col4:
            html_str = m.get_root().render()
            st.download_button("🗺️ Map HTML", data=html_str.encode('utf-8'), file_name="map.html", mime='text/html')

        # Details
        with st.expander("📊 Detailed Statistics"):
            st.dataframe(tap_info, use_container_width=True)

        
if __name__ == '__main__':
    main()
