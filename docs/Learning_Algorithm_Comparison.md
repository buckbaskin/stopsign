# Comparing Learning Algorithms

Originally, the following algorithms were considered for learning which keypoint descriptors describe stopsigns.

  - `GradientBoostingClassifier`
  - `SGDClassifier`
  - `KNeighborsClassifier`
  - `MLPClassifier`
  - `SVC (SupportVectorClassifier)`
  - `DecisionTreeClassifier`

The `GaussianProcessClassifier` was also considered, but it crashed with a MemoryError and was dropped. These methods come directly from scikit-learn and were applied with no constructor arguments.

## The Test

See `classify_descriptors.py`

Using training data subsampled from the actual data with different seeds for the random number generator, each classifier was trained on the subsample for that seed. From there, predictions were made and the accuracy, precision and recall were averaged over the 6 tests.

## The Results

The most accurate model was the `GradientBoostingClassifier` which averaged about 78% accuracy. This model also had the highest precision, but it was below 1%. The model with the highest recall was the `SVC`, which recalled all positive examples, but failed the other two metrics and was removed from consideration. It appears to make poor assumptions for this data set. In practice, the highest recall was the `KNeighborsClassifier` algorithm with about 86% recall.

The algorithm with the best combined precision-recall area was the `KNeighborsClassifier`, followed closely by the `GradientBoostingClassifier`. These two algorithms will be investigated first. The other three algorithms tend to fall short and will be investigated later if need be.

## Further Considerations

For robot saftey, the robot should stop immediately if there is a stopsign visible. With a weaker mandate, the robot shouldn't stop working unless there is a stopsign. In terms of metrics, the robot should aim for high recall for saftey and high precision for allowing continued operation.

Further work may be done to tune the two algorithms and estimate their true precision-recall curve. Additional work may also be done to estimate the presence of a stopsign from multiple keypoints instead of a single keypoint. This can be done using machine learning esimations based on the class labels of all of the keypoints or potentially include a filtering/localization technique such as Kalman filtering to estimate the likelihood and relative position of a stopsign.
