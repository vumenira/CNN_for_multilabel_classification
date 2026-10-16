Macro F1 is appropriate for this competition because it gives equal weight to all five classes. However, validation will likely contain few independent examples of the named animals, making their scores unstable and limiting how confidently we can assess generalization.

An improvement in F1 can also come from adjusting decision thresholds. This changes the precision–recall balance even when the model’s underlying scores remain unchanged.

Macro F1 evaluates labels separately and does not explicitly assess their consistency. For a Moris image, predicting Moris = 1 and Cat = 0 produces a false negative for Cat, but no additional penalty for violating the label hierarchy.

I would also report exact match accuracy: the proportion of images for which the entire predicted label set is correct, including empty sets. This captures correctness at the image level, although frequent cases can dominate the result and hide poor performance on rare classes.

Macro Average Precision  is worth considering as an alternative target metric and a complementary diagnostic. It evaluates ranking quality without choosing a decision threshold, helping distinguish poor class separation from poor threshold selection. It does not solve threshold selection itself: final binary predictions still require thresholds, and estimates for rare classes remain uncertain when validation contains few positives.