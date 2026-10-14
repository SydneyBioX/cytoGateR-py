# API reference

The public API is also available from the flat `cytogater` namespace, for
example `cytogater.run_tree_gating()`.

## Marker-aware gating

```{eval-rst}
.. autofunction:: cytogater.fit_gmm_2
.. autofunction:: cytogater.fit_marker_stats
.. autofunction:: cytogater.gmm_equal_posterior_cutoff
.. autofunction:: cytogater.marker_separability
.. autofunction:: cytogater.score_marker_logistic
.. autofunction:: cytogater.score_marker_rank
.. autofunction:: cytogater.calculate_soft_scores
.. autofunction:: cytogater.assign_soft_labels
.. autofunction:: cytogater.run_soft_gating
.. autofunction:: cytogater.run_tree_gating
.. autofunction:: cytogater.build_fullcoverage_tree
.. autofunction:: cytogater.collect_path_scores
.. autofunction:: cytogater.neg_penalty
.. autofunction:: cytogater.tree_prob
```

## Reference selection and prediction

```{eval-rst}
.. autofunction:: cytogater.apply_cutoff_labels
.. autofunction:: cytogater.assign_confident_labels
.. autofunction:: cytogater.compute_custom_labels
.. autofunction:: cytogater.custom_labels
.. autofunction:: cytogater.prob_mad_cutoff
.. autofunction:: cytogater.prob_quantile_cutoff
.. autofunction:: cytogater.probability_label_matrix
.. autofunction:: cytogater.train_custom_knn
.. autofunction:: cytogater.predict_unknown_with_knn
.. autofunction:: cytogater.predict_wknn_multi
.. autofunction:: cytogater.train_custom_randomforest
.. autofunction:: cytogater.predict_unknown_with_randomforest
.. autofunction:: cytogater.train_custom_dl
.. autofunction:: cytogater.predict_unknown_with_dl
.. autofunction:: cytogater.build_lineage_hierarchy
.. autofunction:: cytogater.build_hierarchical_reference
.. autofunction:: cytogater.predict_hierarchical_knn_recursive
```

## Metrics and uncertainty

```{eval-rst}
.. autofunction:: cytogater.calculate_f1
.. autofunction:: cytogater.calculate_uncertainty
.. autofunction:: cytogater.calculate_spatial_prior_labels
.. autofunction:: cytogater.class_metrics_from_fit
.. autofunction:: cytogater.label_agreement_rates
.. autofunction:: cytogater.label_confusion_matrix
.. autofunction:: cytogater.rf_metric_table
.. autofunction:: cytogater.rf_metric_text
```

## Visualisation

```{eval-rst}
.. autofunction:: cytogater.plot_celltype_tree
.. autofunction:: cytogater.plot_class_metrics
.. autofunction:: cytogater.plot_confusion_matrix
.. autofunction:: cytogater.plot_ct_marker_intensity
.. autofunction:: cytogater.plot_label_agreement
.. autofunction:: cytogater.plot_label_cardinality
.. autofunction:: cytogater.plot_label_confusion_matrix
.. autofunction:: cytogater.plot_label_counts
.. autofunction:: cytogater.plot_label_dotplot
.. autofunction:: cytogater.plot_labelled_cells
.. autofunction:: cytogater.plot_marker_density
.. autofunction:: cytogater.plot_marker_priority_tree
.. autofunction:: cytogater.plot_probability_map
.. autofunction:: cytogater.plot_pseudobulk_heatmap
.. autofunction:: cytogater.plot_rand_cell_probs
.. autofunction:: cytogater.plot_score_hist
.. autofunction:: cytogater.print_celltype_tree
.. autofunction:: cytogater.tree_to_df
```
