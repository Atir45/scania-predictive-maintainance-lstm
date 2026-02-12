# Decision Tree Comparison - Implementation Summary

## What Changed

Your supervisor recommended comparing **different types of decision trees** instead of Random Forest vs XGBoost. This is a better approach because:

1. ✅ All models use the same base algorithm (CART)
2. ✅ Performance differences are directly attributable to configuration choices
3. ✅ Easier to draw clear, meaningful conclusions
4. ✅ More academically rigorous approach

## Files Created/Modified

### New Files Created

1. **`src/models/decision_tree_comparison.py`**
   - Comprehensive decision tree comparison class
   - Trains 8 different decision tree variants
   - Generates detailed comparison metrics and visualizations
   - ~600 lines of production-quality code

2. **`run_decision_tree_analysis.py`**
   - Main execution script
   - Runs complete pipeline with decision tree comparison
   - Saves all results to `results/decision_tree_comparison/`

3. **`docs/decision_tree_comparison_guide.md`**
   - Comprehensive documentation explaining:
     - Each decision tree variant
     - Why we're comparing them
     - How to interpret results
     - How to write your conclusions
     - Academic justification

### Modified Files

1. **`src/scania_pdm_pipeline.py`**
   - Removed Random Forest and XGBoost imports
   - Updated to use Decision Tree variants
   - Modified `train_models()` method to train 6 decision tree variants:
     - Gini (unpruned)
     - Entropy (unpruned)
     - Depth-limited (10, 15)
     - Min-samples constrained
     - Cost-complexity pruned

2. **`README.md`**
   - Updated objectives
   - Added decision tree comparison section
   - Updated quick start guide
   - Updated technologies section

## Decision Tree Variants Being Compared

### 1. Splitting Criteria Comparison
- **Gini Impurity** (Standard CART)
- **Entropy/Information Gain** (Alternative approach)

### 2. Pre-Pruning Strategies
- **Depth-Limited**: max_depth = 10, 15, 20
- **Min-Samples Constrained**: min_samples_split=20, min_samples_leaf=10

### 3. Post-Pruning Strategy
- **Cost-Complexity Pruning**: ccp_alpha = 0.001, 0.0001

## How to Run

```bash
# Run the complete analysis
python run_decision_tree_analysis.py
```

This will:
1. Load and preprocess data
2. Engineer features
3. Train all 8 decision tree variants
4. Evaluate on validation set
5. Generate comprehensive visualizations
6. Save results to `results/decision_tree_comparison/`

## Generated Outputs

### Metrics & Reports
- `model_comparison.csv` - Complete comparison table
- `detailed_results.json` - Full results with confusion matrices
- `comparison_report.txt` - Human-readable summary

### Visualizations
- `metrics_comparison.png` - 4-panel: Accuracy, Precision, Recall, F1
- `roc_auc_comparison.png` - ROC-AUC scores across models
- `tree_structure_comparison.png` - Nodes, Depth, Leaves comparison
- `top_models_confusion_matrix.png` - Confusion matrices for best 3 models
- `feature_importance_comparison.png` - Feature importance for top models

## For Your Report/Thesis

### Introduction Section
"This study compares eight decision tree variants to identify the optimal configuration for predictive maintenance of Scania truck components. All variants use the Classification and Regression Trees (CART) algorithm but differ in splitting criteria (Gini vs Entropy) and pruning strategies (pre-pruning via depth/sample constraints vs post-pruning via cost-complexity)."

### Methodology Section
"We systematically investigate:
1. **Splitting Criterion Effect**: Gini impurity vs Entropy/Information Gain
2. **Pre-Pruning Strategies**: Depth limitation and minimum sample constraints
3. **Post-Pruning Approach**: Cost-complexity pruning with varying α values"

### Results Section
Use the generated comparison table and visualizations to show:
- Performance metrics across all models
- Trade-offs between model complexity and performance
- Optimal configuration for your specific problem

### Discussion Section Template
```
The comparison revealed several key findings:

1. Splitting Criterion: Gini and Entropy trees showed similar performance 
   (F1: 0.XX vs 0.XX), confirming that criterion choice has minimal impact 
   on final predictions for this dataset.

2. Pre-Pruning by Depth: Depth-limited trees demonstrated improved 
   generalization, with optimal performance at depth=15 (F1: 0.XX), 
   balancing model complexity and predictive power.

3. Post-Pruning: Cost-complexity pruning achieved the best balance, 
   producing a tree with XXX nodes (vs XXXX for unpruned) while 
   maintaining F1=0.XX.

4. Overfitting Evidence: Unpruned trees showed XXX nodes, suggesting 
   potential overfitting, while pruned variants generalized better.
```

### Conclusions Section
"Among the eight decision tree variants tested:
- **Best Overall Performance**: [Model Name] (F1=0.XX, ROC-AUC=0.XX)
- **Best Generalization**: [Model Name] with balanced complexity/performance
- **Recommendation**: [Model Name] for deployment due to [reasons]

This systematic comparison demonstrates that [pruning strategy] is most 
effective for this predictive maintenance task, providing both interpretability 
and robust performance."

## Key Advantages of This Approach

✅ **Fair Comparison**: All models use same base algorithm  
✅ **Clear Attribution**: Performance differences due to specific choices  
✅ **Academic Rigor**: Systematic parameter study  
✅ **Interpretable**: Each tree is fully explainable  
✅ **Practical**: Identifies optimal configuration for your problem  

## Next Steps

1. **Run the analysis**: 
   ```bash
   python run_decision_tree_analysis.py
   ```

2. **Review results**: Check `results/decision_tree_comparison/`

3. **Identify best model**: Based on F1 score and business requirements

4. **Write findings**: Use the comparison guide to structure your discussion

5. **Optional enhancements**:
   - Cross-validation for robustness
   - Learning curves
   - Cost-sensitive analysis (weight FN vs FP differently)
   - Feature importance deep-dive

## Questions for Your Supervisor

After running the analysis, discuss:
1. "Which metric should we prioritize - F1 score or recall given the safety implications?"
2. "The optimal tree depth appears to be around X levels. Does this align with domain expectations?"
3. "Should we investigate cost-sensitive learning to weight false negatives more heavily?"

## Support

See detailed documentation in:
- `docs/decision_tree_comparison_guide.md` - Complete guide with theory and interpretation
- `README.md` - Updated project overview
- Code comments in `src/models/decision_tree_comparison.py`

---

**Implementation Date**: November 2025  
**Status**: ✅ Complete and ready to run
