# Tap Placement Optimizer - Complete User Guide
**Version 3.0 - Comprehensive Edition**

---

## QUICK NAVIGATION

- [Overall Score - THE Critical Metric](#overall-score)
- [All Metrics Explained](#all-metrics)
- [Voronoi Boundaries](#voronoi)
- [Workflow Guide](#workflow)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

<a name="overall-score"></a>
# OVERALL SCORE - THE CRITICAL METRIC

The **Overall Score (0-100)** is the single most important metric. It tells you if your tap configuration is ready to implement.

## The Formula

```
Overall Score = (Distance Score × 40%) + (Equity Score × 30%) + (Coverage Score × 30%)
```

### Why These Weights?
- **Distance (40%)** = MOST IMPORTANT - affects daily life
- **Equity (30%)** = Fair service - no overcrowded taps
- **Coverage (30%)** = International standard (70%+ within 250m)

---

## Component 1: Distance Score (40%)

**Formula:**
```python
distance_score = 100 - (avg_distance / 500 × 100)
```

**Every meter costs 0.2 points**

### Examples:

| Avg Distance | Score | Contribution | Quality |
|--------------|-------|--------------|---------|
| 75m | 85 | 34 points | ⭐⭐⭐⭐⭐ Excellent |
| 125m | 75 | 30 points | ⭐⭐⭐⭐ Very Good |
| 187m | 62.6 | 25 points | ⭐⭐⭐ Good |
| 250m | 50 | 20 points | ⭐⭐ Acceptable |
| 400m | 20 | 8 points | ⭐ Poor |

### Real Impact:
- **<150m**: Excellent - 2-3 min walk, easy collection
- **150-250m**: Good - 5 min walk, acceptable burden
- **250-400m**: Poor - 8-10 min walk, significant burden
- **>400m**: Critical - major daily life impact

---

## Component 2: Equity Score (30%)

**Formula:**
```python
raw_equity = (std_dev / mean) × 100
normalized = 100 - raw_equity
```

**Lower raw equity = better** (more balanced loads)

### Examples:

**Perfect Balance:**
```
Taps: [40, 40, 40, 40, 40]
Raw equity: 0 → Normalized: 100
Contributes: 30 points ✅
```

**Excellent Balance:**
```
Taps: [38, 42, 35, 40, 45, 37, 41, 39]
Raw equity: 7.8 → Normalized: 92.2
Contributes: 27.7 points ✅
```

**Poor Balance:**
```
Taps: [20, 40, 60, 50, 80]
Raw equity: 50 → Normalized: 50
Contributes: 15 points ❌
```

### Interpretation:

| Raw Equity | Normalized | Quality |
|------------|------------|---------|
| 0-10 | 90-100 | ✅ Excellent |
| 10-20 | 80-90 | ✅ Good |
| 20-30 | 70-80 | ⚠️ Acceptable |
| 30-50 | 50-70 | ⚠️ Fair |
| >50 | <50 | ❌ Poor |

---

## Component 3: Coverage Score (30%)

**Formula:**
```python
coverage_score = (households_within_250m / total_households) × 100
```

Simple percentage - direct measure of accessibility.

### Examples:

| Coverage | Contribution | Quality |
|----------|--------------|---------|
| 90% | 27 points | ⭐⭐⭐⭐⭐ Excellent |
| 80% | 24 points | ⭐⭐⭐⭐ Very Good |
| 70% | 21 points | ⭐⭐⭐ Good |
| 60% | 18 points | ⭐⭐ Fair |
| 50% | 15 points | ⭐ Poor |

---

## Complete Example Calculation

**Scenario:** 450 households, 12 taps

**Given:**
- Average distance: 187m
- Tap loads: [38, 42, 35, 40, 45, 37, 41, 39, 43, 36, 44, 40]
- Within 250m: 315 of 450

**Calculation:**

1. **Distance Score:** 100 - (187/500 × 100) = 62.6 → **25.0 points**
2. **Equity Score:** Raw 7.8 → Normalized 92.2 → **27.7 points**
3. **Coverage Score:** 315/450 = 70% → **21.0 points**

**Overall:** 25.0 + 27.7 + 21.0 = **73.7 ≈ 74/100**

**Rating:** ⭐⭐⭐ GOOD

**Interpretation:**
- ✅ Excellent balance (92.2)
- ✅ Meets coverage standard (70%)
- ⚠️ Moderate distances (62.6) - could improve

**Decision:** Can implement - meets minimum standards

---

## Rating Scale

### ⭐⭐⭐⭐⭐ EXCELLENT (90-100)
- Outstanding configuration
- Exceeds all standards
- Ready for immediate implementation

**To Achieve 90:**
```
Distance 75m (85 points) → 34.0
Equity raw 5 (95 points) → 28.5
Coverage 90% → 27.0
Total: 89.5
```

---

### ⭐⭐⭐⭐ VERY GOOD (80-89)
- Strong performance
- Exceeds minimum standards
- Ready for implementation

**To Achieve 85:**
```
Distance 125m (75 points) → 30.0
Equity raw 8 (92 points) → 27.6
Coverage 85% → 25.5
Total: 83.1
```

---

### ⭐⭐⭐ GOOD (70-79) ← **MINIMUM ACCEPTABLE**
- Meets minimum standards
- Acceptable performance
- Can implement but monitor

**Minimum for Implementation:**
```
Distance 200m (60 points) → 24.0
Equity raw 20 (80 points) → 24.0
Coverage 70% → 21.0
Total: 69.0 ≈ 70
```

---

### ⭐⭐ FAIR (60-69) ← **NEEDS IMPROVEMENT**
- Below standards
- Should NOT implement
- Redesign recommended

---

### ⭐ POOR (<60) ← **UNACCEPTABLE**
- Significant problems
- Cannot implement
- Major redesign required

---

## Decision Guide

| Score | Can Implement? | Action |
|-------|----------------|--------|
| 90-100 | ✅ YES | Proceed immediately |
| 80-89 | ✅ YES | Proceed confidently |
| 75-79 | ✅ YES | Proceed with confidence |
| 70-74 | ✅ YES | Proceed, monitor closely |
| 60-69 | ❌ NO | Improve before implementation |
| <60 | ❌ NO | Major redesign required |

---

## Critical Thresholds - NEVER ACCEPT

Reject configuration if ANY of these are true:

- ❌ Average distance >400m
- ❌ Any household >500m
- ❌ Coverage <50% at 250m
- ❌ Any tap >80 households
- ❌ Equity raw >50

These are **hard limits** regardless of overall score.

---

<a name="all-metrics"></a>
# ALL METRICS EXPLAINED

## Distance Metrics

**Average Distance**
- Mean walking distance to nearest tap
- Target: <250m (optimal), <400m (acceptable)
- Key predictor of water consumption

**Median Distance**
- Middle value when sorted
- Shows "typical" household experience
- Compare to average to understand distribution

**Max Distance**
- Furthest household
- Target: <500m (critical)
- Identifies most underserved

---

## Coverage Metrics

**Coverage at 250m**
- % within optimal distance
- Target: >70% (eWater standard)
- International WASH benchmark

**Coverage at 400m**
- % within acceptable distance
- Target: >95%
- Critical threshold

**Coverage at 500m**
- % within absolute limit
- Target: 100%
- If <100% = emergency

---

## Load Balance Metrics

**Min/Max/Avg Households per Tap**
- Lightest/heaviest/average load
- Target: 30-50 HH per tap
- Used for capacity planning

**Equity Score**
- Variation in tap loads
- Lower = more balanced
- Target: <30 (raw score)

---

## Distance Bins

11 bins categorize every household:

| Bin | Distance | Quality | Color |
|-----|----------|---------|-------|
| <50m | 0-49m | ⭐⭐⭐⭐⭐ Exceptional | Bright Green |
| 50-99m | 50-99m | ⭐⭐⭐⭐⭐ Excellent | Bright Green |
| 100-149m | 100-149m | ⭐⭐⭐⭐ Very Good | Light Green |
| 150-199m | 150-199m | ⭐⭐⭐⭐ Good | Light Green |
| 200-249m | 200-249m | ⭐⭐⭐ Good | Light Green |
| 250-299m | 250-299m | ⭐⭐ Acceptable | Orange |
| 300-349m | 300-349m | ⭐⭐ Acceptable | Orange |
| 350-399m | 350-399m | ⭐ Marginal | Orange |
| 400-449m | 400-449m | ⚠️ Poor | Red |
| 450-499m | 450-499m | ⚠️ Very Poor | Red |
| >500m | 500+m | 🚫 Critical | Dark Red |

**Ideal:** Most households in green bins (<250m)

---

<a name="voronoi"></a>
# VORONOI BOUNDARIES

Voronoi diagrams show service area boundaries.

## What Are They?

Mathematical polygons where each represents the area closest to one tap than any other.

## 7 Critical Uses

### 1. Visual Service Area Mapping
See which geographic area each tap serves

### 2. Identify Coverage Gaps
Large polygons = underserved regions

### 3. Balance Service Loads
Compare polygon sizes to tap loads

### 4. Optimize Tap Positions
Check if tap is centered in its polygon

### 5. Field Deployment Planning
Assign teams by Voronoi region

### 6. Spot Territorial Conflicts
Check if boundaries cross rivers, roads, borders

### 7. Verify Optimization
Visual check algorithm worked correctly

## Requirements
- Need ≥4 taps
- scipy and shapely libraries
- Valid coordinates

## How to Read

**Good (Uniform):**
```
Similar-sized polygons
Taps centered
Smooth boundaries
```

**Problem (Stretched):**
```
One huge polygon = gap
Very small polygon = underutilized
Jagged boundaries = issues
```

---

<a name="workflow"></a>
# WORKFLOW GUIDE

## Case 1: New Installation (8 Steps)

### 1. Upload Household Data
CSV with `SM Latitude`, `SM Longitude`

### 2. Get Smart Recommendations
Click "🔮 Get Smart Recommendations"
App suggests optimal parameters

### 3. Apply or Adjust
Apply recommendations OR manually adjust sliders

### 4. Propose Taps
Click "🎯 Propose Optimal Tap Locations"
K-means algorithm runs

### 5. Review Results
- Overall score (target ≥70)
- Metrics
- Map & Voronoi
- Charts
- Recommendations

### 6. Download Package
- Tap Locations CSV (GPS coordinates)
- Tap Statistics CSV (performance)
- Household Assignments CSV
- Map HTML (offline viewing)

### 7. Field Adjustment
- Navigate to proposed locations
- Assess suitability
- Adjust if needed (±50m acceptable)
- Document changes

### 8. Re-upload for Final Evaluation
- Upload adjusted coordinates
- Verify score still ≥70
- Final approval

---

## Case 2: Evaluating Existing Taps

### 1. Upload Both Files
- Households CSV
- Existing taps CSV

### 2. Auto-Evaluation
Happens automatically - no button needed

### 3. Review Performance
Check score and metrics

### 4. Decision
- Score ≥80: Excellent, proceed
- Score 70-79: Acceptable, proceed
- Score 60-69: Improve first
- Score <60: Redesign required

---

<a name="best-practices"></a>
# BEST PRACTICES

## Parameter Selection

### Number of Taps
```
Rural: 1 per 30-40 HH
Suburban: 1 per 40-60 HH
Urban: 1 per 60-80 HH

Example: 450 HH rural → 11-15 taps
```

### Min Households
```
Conservative: 20-25 (more taps)
Moderate: 25-35 (balanced)
Aggressive: 35-45 (fewer taps)
```

### Max Households
```
Optimal: 60-70
Acceptable: 70-80
Avoid: >80 (congestion)
```

## Field Adjustments

**When to adjust:**
- Private property
- Blocked access
- Better location nearby
- Community preference

**How much:**
- Keep within 50m: OK
- 50-100m: Document reason
- >100m: Re-optimize needed

**Document:**
- Photo (4 directions)
- Adjusted GPS
- Reason
- Community approval

---

<a name="troubleshooting"></a>
# TROUBLESHOOTING

## Low Score Despite Many Taps

**Cause:** Geographic constraints (very dispersed)

**Solution:**
- Add more taps
- Accept geographic reality
- Hybrid approach (community + household taps)
- Phased implementation

---

## High Equity Score (Imbalance)

**Cause:** Uneven distribution or poor placement

**Solution:**
- Increase number of taps
- Manually adjust positions
- Accept some imbalance if justified
- Remove underutilized taps

---

## All Taps Removed Error

**Cause:** Conflicting parameters

**Solution:**
- Reduce number of taps
- Lower min households
- Use recommendations
- Rule: Min HH < (Total HH ÷ Num Taps)

---

## Voronoi Not Showing

**Causes & Solutions:**
- <4 taps: Add more
- Libraries missing: Check installation
- Invalid coordinates: Validate data
- Browser issue: Refresh, try different browser

---

# QUICK REFERENCE

## Critical Numbers
- **70** = Minimum acceptable overall score
- **250m** = Optimal distance (WHO standard)
- **400m** = Maximum acceptable distance
- **500m** = Absolute limit (never exceed)
- **70%** = Minimum coverage at 250m
- **30-60 HH** = Optimal tap load range
- **<30** = Target equity score (raw)

## Component Weights
- Distance: **40%** (most important)
- Equity: **30%** (fairness)
- Coverage: **30%** (standard)

## Decision Thresholds
- **90+**: Excellent, implement immediately
- **80-89**: Very good, implement
- **70-79**: Good, minimum acceptable
- **60-69**: Fair, improve first
- **<60**: Poor, redesign required

---

# DOWNLOADS EXPLAINED

## 1. Tap Statistics CSV
- Per-tap performance metrics
- Distance breakdown by bins
- Use for field assignments, reporting

## 2. Tap Locations CSV
- GPS coordinates for navigation
- Import to GPS device
- Field team reference

## 3. Household Assignments CSV
- Which HH assigned to which tap
- Use for community sensitization
- Billing setup

## 4. Map HTML
- Interactive offline map
- All features included
- Presentations, field reference

---

# SCORE EXAMPLES

## Example 1: Excellent (92/100)
```
Distance: 95m → Score 81 → 32.4
Equity: Raw 6 → Normalized 94 → 28.2
Coverage: 88% → 26.4
Total: 86.6
```
**Ready for immediate implementation**

---

## Example 2: Good (74/100)
```
Distance: 187m → Score 62.6 → 25.0
Equity: Raw 7.8 → Normalized 92.2 → 27.7
Coverage: 70% → 21.0
Total: 73.7
```
**Acceptable, can implement**

---

## Example 3: Fair (65/100)
```
Distance: 250m → Score 50 → 20.0
Equity: Raw 20 → Normalized 80 → 24.0
Coverage: 65% → 19.5
Total: 63.5
```
**Below standard, improve first**

---

## Example 4: Poor (57/100)
```
Distance: 300m → Score 40 → 16.0
Equity: Raw 35 → Normalized 65 → 19.5
Coverage: 60% → 18.0
Total: 53.5
```
**Unacceptable, major redesign required**

---

This complete guide provides everything you need to understand, use, and optimize the Tap Placement Optimizer! 🎯
