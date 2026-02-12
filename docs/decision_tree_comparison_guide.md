# Decision Tree Comparison Study
## Scania Predictive Maintenance Project

### Overview
This document explains the different decision tree variants compared in this study and the rationale for comparing them.

---

## Why Compare Different Decision Tree Variants?

Your supervisor's recommendation to compare different types of decision trees (rather than Random Forest vs XGBoost) is excellent for several reasons:

1. **Fair Comparison**: All variants use the same base algorithm (CART - Classification and Regression Trees)
2. **Direct Attribution**: Performance differences can be directly attributed to specific algorithmic choices (criterion, pruning strategy, etc.)
3. **Clear Conclusions**: Easier to draw meaningful conclusions about which approach works best for your specific problem
4. **Academic Rigor**: Demonstrates systematic investigation of a single algorithm family with different configurations

---

## Decision Tree Variants Compared

### 1. **Gini Impurity (Unpruned)**
- **Criterion**: Gini impurity
- **Configuration**: No pruning constraints (max_depth=None)
- **Description**: Standard CART algorithm using Gini impurity to measure split quality
- **Formula**: Gini = 1 - Σ(p_i²) where p_i is proportion of class i
- **Use Case**: Baseline unpruned tree using most common splitting criterion

### 2. **Entropy/Information Gain (Unpruned)**
- **Criterion**: Entropy (Information Gain)
- **Configuration**: No pruning constraints (max_depth=None)
- **Description**: Alternative splitting criterion based on information theory
- **Formula**: Entropy = -Σ(p_i * log₂(p_i))
- **Use Case**: Compare information-theoretic approach vs probabilistic approach (Gini)

### 3. **Depth-Limited Trees (Pre-Pruning)**
- **Variants**: max_depth = 10, 15, 20
- **Description**: Prevents trees from growing beyond specified depth
- **Purpose**: Control model complexity and prevent overfitting
- **Trade-off**: Simpler models (less overfitting) vs potential underfitting

### 4. **Min-Samples Constrained (Pre-Pruning)**
- **Configuration**: 
  - min_samples_split = 20 (require 20+ samples to split node)
  - min_samples_leaf = 10 (require 10+ samples in leaf node)
- **Description**: Prunes tree by requiring minimum samples at nodes
- **Purpose**: Ensure statistical significance of splits
- **Benefits**: More robust to noise, better generalization

### 5. **Cost-Complexity Pruning (Post-Pruning)**
- **Variants**: ccp_alpha = 0.001, 0.0001
- **Description**: Post-pruning using cost-complexity criterion
- **Algorithm**: 
  1. Grow full tree
  2. Calculate cost-complexity for each subtree
  3. Prune subtrees that don't improve cost-complexity
- **Parameter**: ccp_alpha controls pruning aggressiveness (higher = more pruning)
- **Advantage**: More principled than pre-pruning, considers entire tree structure

---

## Comparison Dimensions

### Performance Metrics
1. **Accuracy**: Overall correctness
2. **Precision**: Of predicted failures, how many were actual failures?
3. **Recall**: Of actual failures, how many did we catch?
4. **F1 Score**: Harmonic mean of precision and recall
5. **ROC-AUC**: Discrimination ability across all thresholds

### Tree Structure Metrics
1. **Number of Nodes**: Total nodes in tree (complexity measure)
2. **Maximum Depth**: How deep the tree grows
3. **Number of Leaves**: Terminal nodes (decision points)

### Business Impact (for Predictive Maintenance)
1. **False Negatives (FN)**: Missed failures - HIGH COST (unexpected breakdowns)
2. **False Positives (FP)**: Unnecessary maintenance - MEDIUM COST (wasted resources)
3. **True Positives (TP)**: Caught failures - VALUE (prevented breakdowns)

---

## Expected Findings & Discussion Points

### Unpruned Trees (Gini vs Entropy)
- **Expected**: Similar performance, Gini slightly faster to compute
- **Discussion**: "Both unpruned trees likely show high recall but may overfit training data..."

### Pre-Pruning (Depth-Limited)
- **Expected**: Better generalization than unpruned, performance depends on optimal depth
- **Discussion**: "Depth-limited trees show the trade-off between model complexity and generalization..."

### Pre-Pruning (Min-Samples)
- **Expected**: More robust to noise, potentially better on validation data
- **Discussion**: "Requiring minimum samples ensures statistical reliability of splits..."

### Post-Pruning (Cost-Complexity)
- **Expected**: Best balance between complexity and performance
- **Discussion**: "Cost-complexity pruning provides a principled approach to removing unnecessary complexity..."

---

## How to Interpret Results

### 1. Performance vs Complexity Trade-off
```
High Complexity (many nodes) + High Performance = Potential Overfitting
Low Complexity (few nodes) + High Performance = Good Generalization
Low Complexity + Low Performance = Underfitting
```

### 2. For Predictive Maintenance
- **Priority**: High Recall (catch failures before they happen)
- **Secondary**: Reasonable Precision (don't waste too much on unnecessary maintenance)
- **Goal**: Best F1 score while maintaining high recall

### 3. Model Selection Criteria
Choose model that:
1. Has highest F1 score on validation data
2. Shows reasonable complexity (not overfitted)
3. Maintains acceptable recall (>70-80% for safety-critical applications)

---

## Writing Your Comparison & Conclusions

### Structure for Your Report/Thesis

#### 1. Introduction
"This study compares six decision tree variants to identify the optimal configuration for predictive maintenance of Scania truck components. All variants use the CART algorithm but differ in splitting criteria and pruning strategies."

#### 2. Methodology
"We compare:
- Splitting criteria: Gini impurity vs Entropy
- Pruning approaches: Pre-pruning (depth-limited, min-samples) vs Post-pruning (cost-complexity)
- Model complexity: Unpruned vs various pruning configurations"

#### 3. Results
Present the comparison table and visualizations showing:
- Performance metrics across all models
- Tree structure characteristics
- Confusion matrices for top performers

#### 4. Discussion
Example structure:
- "The unpruned Gini and Entropy trees showed similar performance (F1: 0.XX vs 0.XX), confirming that splitting criterion choice has minimal impact..."
- "Depth-limited trees demonstrated improved generalization, with optimal performance at depth=15 (F1: 0.XX)..."
- "Cost-complexity pruning achieved the best balance between model complexity (XXX nodes) and performance (F1: 0.XX)..."

#### 5. Conclusions
"Among the six decision tree variants tested:
1. Best Overall: [Model Name] achieved F1=0.XX with ROC-AUC=0.XX
2. Most Generalizable: [Model Name] showed best validation performance
3. Recommendation: [Model Name] is recommended for deployment due to [reasons]"

### Key Advantages of This Approach
✓ All models use same algorithm family (fair comparison)
✓ Performance differences directly attributable to configuration choices
✓ Clear narrative: "investigated effect of splitting criteria and pruning strategies"
✓ Academically rigorous: systematic parameter study
✓ Practical insights: identifies optimal configuration for your specific problem

---

## Running the Analysis

```bash
# Run the complete decision tree comparison
python run_decision_tree_analysis.py
```

This will:
1. Load and preprocess data
2. Train all 8 decision tree variants
3. Evaluate on validation set
4. Generate comparison visualizations
5. Save detailed results to `results/decision_tree_comparison/`

---

## Generated Outputs

### CSV Files
- `model_comparison.csv` - Complete metrics for all models

### JSON Files
- `detailed_results.json` - Full results including confusion matrices

### Visualizations
- `metrics_comparison.png` - Accuracy, Precision, Recall, F1
- `roc_auc_comparison.png` - ROC-AUC scores
- `tree_structure_comparison.png` - Nodes, Depth, Leaves
- `top_models_confusion_matrix.png` - Best 3 models
- `feature_importance_comparison.png` - Top features

### Reports
- `comparison_report.txt` - Human-readable summary

---

## Next Steps

1. **Run the analysis**: Execute `run_decision_tree_analysis.py`
2. **Review results**: Check generated visualizations and reports
3. **Select best model**: Based on F1 score and business requirements
4. **Write findings**: Use results to support your discussion/conclusions
5. **Further analysis** (optional):
   - Feature importance analysis
   - Learning curves
   - Cross-validation for robustness

---

## Questions for Your Supervisor

Based on results, you might discuss:
1. "The cost-complexity pruned tree achieved best F1 (0.XX). Is this suitable for deployment?"
2. "Should we prioritize recall over precision given the high cost of unexpected failures?"
3. "The optimal depth appears to be around 15 levels. Does this align with domain expectations?"

---

## References

- Breiman, L., et al. (1984). "Classification and Regression Trees" (CART algorithm)
- Quinlan, J.R. (1986). "Induction of Decision Trees" (ID3/C4.5 algorithms)
- Scikit-learn Documentation: Decision Trees
  https://scikit-learn.org/stable/modules/tree.html

---

**Document Version**: 1.0  
**Last Updated**: November 2025  
**Author**: ML Engineer
