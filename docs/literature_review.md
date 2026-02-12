# Literature Review: Scania Component X Dataset

## Dataset Information

**Official Name:** SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive Maintenance

**Version:** Version 2 (used in this project)
- Note: Version 3 is now available (2025-04-01)

**Source:** Swedish National Data Service (SND)
- DOI: https://doi.org/10.5878/jvb5-d390
- URL: https://researchdata.se/sv/catalogue/dataset/2024-34/2

**Authors:**
- Tony Lindgren (Stockholm University, DSV)
- Olof Steinert (Scania CV AB)
- Oskar Andersson Reyna (Scania CV AB)
- Zahra Kharazian (Stockholm University, DSV)
- Sindri Magnússon (Stockholm University, DSV)

**Published:** July 3, 2024
**License:** CC BY 4.0

---

## Primary Research Papers

### 1. Main Dataset Paper (PRIMARY BASELINE)
**Title:** "SCANIA component X dataset: a real-world multivariate time series dataset for predictive maintenance"

**Authors:** Kharazian, Z., Lindgren, T., Magnússon, S., Steinert, O., et al.

**Published:** Scientific Data (Nature), 2025

**Citations:** 21 citations (as of search date)

**Access:**
- Nature: https://www.nature.com/articles/s41597-025-04802-6
- ResearchGate: [PDF available]

**Importance:** ⭐⭐⭐⭐⭐
- This is THE baseline paper for your dataset
- Contains official methodology and benchmarks
- Published in high-impact journal (Scientific Data - Nature)
- Most cited work on this dataset

**Action Items:**
- [ ] Download and read full paper
- [ ] Extract their preprocessing methodology
- [ ] Note their feature engineering approach
- [ ] Record their baseline model results
- [ ] Compare your custom decision tree to their benchmarks

---

### 2. SHAP-Driven Explainability Paper
**Title:** "SHAP-Driven Explainability in Survival Analysis for Predictive Maintenance Applications"

**Authors:** Kharazian, Z., Miliou, I., Lindgren, T., et al.

**Published:** 2024

**Focus:** Survival analysis approach to predictive maintenance on Scania Component X

**Relevance:** ⭐⭐⭐
- Uses same dataset
- Different modeling approach (survival analysis vs classification)
- Includes explainability methods (SHAP values)

---

### 3. Conformal Prediction Paper
**Title:** "Copal: Conformal prediction for active learning with application to remaining useful life estimation in predictive maintenance"

**Authors:** Kharazian, Z., Lindgren, T., et al.

**Published:** Proceedings of Machine Learning Research, 2024

**Citations:** 2

**Focus:** Active learning + conformal prediction for RUL estimation

**Relevance:** ⭐⭐
- Uses Component X dataset
- Different approach (active learning)
- May have useful preprocessing insights

---

### 4. Synthetic Data Generation Paper
**Title:** "Low dimensional synthetic data generation for improving data driven prognostic models"

**Authors:** Lindgren, T., Steinert, O.

**Published:** IEEE International Conference, 2022

**Citations:** 5

**Focus:** Using synthetic data to improve prognostic models on Scania data

**Relevance:** ⭐⭐
- Early work with similar dataset
- Addresses class imbalance with synthetic data
- May have baseline results

---

### 5. Contrastive Learning Paper
**Title:** "Robust contrastive learning and multi-shot voting for high-dimensional multivariate data-driven prognostics"

**Authors:** Steinert, O., Lindgren, T., et al.

**Published:** IEEE International Conference, 2023

**Citations:** 4

**Focus:** Contrastive learning for high-dimensional predictive maintenance

**Relevance:** ⭐⭐⭐
- Uses Scania heavy truck data
- Addresses high-dimensionality challenge
- May have feature selection insights

---

### 6. Explainable Anomaly Detection (Thesis)
**Title:** "Explainable Anomaly Detection in Predictive Maintenance Using Shapelet Transform"

**Author:** San, W.

**Published:** 2024 (Master's Thesis)

**Supervisor:** Tony Lindgren

**Focus:** Anomaly detection using shapelet transform on Scania data

**Relevance:** ⭐⭐
- Student work using same dataset
- Different approach (anomaly detection)
- May have simpler baseline models

---

## Key Research Directions from Literature

1. **Survival Analysis Approach**
   - Time-to-failure prediction
   - Censored data handling
   - May be more appropriate than binary classification

2. **Class Imbalance Handling**
   - Synthetic data generation (SMOTE, GANs)
   - Cost-sensitive learning
   - Class weighting (you're already doing this!)

3. **High-Dimensionality**
   - Feature selection (mutual information)
   - Dimensionality reduction
   - Contrastive learning

4. **Explainability**
   - SHAP values
   - Feature importance
   - Shapelet transforms

5. **Active Learning**
   - Selective labeling
   - Conformal prediction
   - Uncertainty quantification

---

## Next Steps for Your Project

### Immediate Actions:

1. **Get the Primary Paper**
   - Contact: tony@dsv.su.se (Tony Lindgren)
   - Or check your university library for Nature Scientific Data access
   - Extract their exact methodology

2. **Replicate Their Preprocessing**
   - Use sklearn (as Felix suggested)
   - Match their feature engineering
   - This ensures fair comparison

3. **Implement Your Custom Decision Tree**
   - Focus on the model itself
   - Add class weighting (done ✅)
   - Consider adding:
     - Entropy criterion
     - Pruning
     - Feature importance

4. **Compare Results**
   - Your custom tree vs their baseline
   - Your custom tree vs sklearn
   - Report metrics they used in paper

### Research Questions to Answer:

- What preprocessing did they use?
- What features did they engineer?
- What models did they benchmark?
- What metrics did they report?
- What were their baseline results?
- How did they handle class imbalance?

---

## Contact Information

**Primary Researcher:** Tony Lindgren
- Email: tony@dsv.su.se
- Affiliation: Stockholm University, Department of Computer and Systems Sciences (DSV)
- ORCID: 0000-0001-7713-1381

**Industry Partner:** Olof Steinert
- Affiliation: Scania CV AB, Strategic Product Planning and Advanced Analytics
- ORCID: 0000-0001-7750-XXXX (check paper)

---

## Notes

- The dataset is specifically designed for benchmarking predictive maintenance algorithms
- Multiple approaches have been tried: classification, survival analysis, anomaly detection
- Class imbalance is a known challenge (acknowledged in multiple papers)
- High dimensionality (105 sensors) is another key challenge
- The dataset is actively used in research (21 citations in ~1 year)

---

## References Format

**Main Citation:**
```
Lindgren, T., Steinert, O., Andersson Reyna, O., Kharazian, Z., & Magnússon, S. (2024). 
SCANIA Component X Dataset: A Real-World Multivariate Time Series Dataset for Predictive Maintenance 
(Version 2) [Data set]. Scania CV AB. https://doi.org/10.5878/jvb5-d390
```

**Main Paper Citation:**
```
Kharazian, Z., Lindgren, T., Magnússon, S., Steinert, O., et al. (2025). 
SCANIA component X dataset: a real-world multivariate time series dataset for predictive maintenance. 
Scientific Data, Nature.
```
