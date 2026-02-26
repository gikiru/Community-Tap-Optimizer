# 🎯 Community Tap Optimizer (v3.0)

**Data-driven geographical modeling for optimized water infrastructure deployment.**

The **Community Tap Optimizer** is a specialized spatial analysis tool designed for the WASH (Water, Sanitation, and Hygiene) sector. It enables water utilities and social enterprises to model and evaluate community tap locations based on household geographical data, ensuring equitable and efficient water access.

## 🚀 Key Features

- **Geospatial Optimization:** Uses K-Means clustering and Voronoi tessellation to propose optimal tap locations.
- **M&E Metrics:** Automatically calculates an **Overall Performance Score (0-100)** based on:
  - **Distance (40%):** Proximity of households to taps.
  - **Equity (30%):** Fairness of household distribution per tap.
  - **Coverage (30%):** Adherence to international standards (e.g., eWater, WHO).
- **Interactive Visualization:** Generates offline-capable HTML maps with Voronoi boundaries and service area overlays.
- **Field-Ready Exports:** Generates GPS-ready coordinates (CSV) and household assignment logs for community implementation.

## 🛠 Tech Stack

- **Core:** Python 3.x
- **Spatial Analysis:** , , 
- **Visualization:**  (Maps),  (Analytics)
- **Deployment:** Pre-configured batch scripts (, ) for Windows environments.

## 📈 Impact-Driven Design

In many rural and peri-urban water projects, tap placement is often arbitrary, leading to distance inequities and system overloads. This tool replaces "best-guessing" with a rigorous mathematical framework, ensuring that at least **70% of households are within 250m** of a water point while maintaining balanced tap loads.

---

### 📖 Documentation & Guides

For a deep dive into the scoring formulas, technical metrics, and field deployment workflows, please see the **[Tap Optimizer Guide.md](./Tap%20Optimizer%20Guide.md)**.

---