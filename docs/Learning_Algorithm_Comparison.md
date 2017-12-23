# Comparing Learning Algorithms

Originally, the following algorithms were considered for learning which keypoint descriptors describe stopsigns.

  - `GradientBoostingClassifier`
  - `SGDClassifier`
  - `KNeighborsClassifier`
  - `MLPClassifier`
  - `SVC (SupportVectorClassifier)`
  - `DecisionTreeClassifier`

The `GaussianProcessClassifier` was also considered, but it crashed with a MemoryError and was dropped. These methods come directly from scikit-learn and were applied with no constructor arguments.

## Round 1: The Test

See `classify_descriptors.py`

Using training data subsampled from the actual data with different seeds for the random number generator, each classifier was trained on the subsample for that seed. From there, predictions were made and the accuracy, precision and recall were averaged over the 6 tests.

## Round 1: The Results

The most accurate model was the `GradientBoostingClassifier` which averaged about 78% accuracy. This model also had the highest precision, but it was below 1%. The model with the highest recall was the `SVC`, which recalled all positive examples, but failed the other two metrics and was removed from consideration. It appears to make poor assumptions for this data set. In practice, the highest recall was the `KNeighborsClassifier` algorithm with about 86% recall.

The algorithm with the best combined precision-recall area was the `KNeighborsClassifier`, followed closely by the `GradientBoostingClassifier`. These two algorithms will be investigated first. The other three algorithms tend to fall short and will be investigated later if need be.

## Round 2: The Test

See `ml_compare.py`

Using existing data, train each algorithm a bunch of times with different initialization parameters. Additionally, the prediction time is considered. Each algorithm is run 10 times with different seeds and then averaged. This averaging still leads to some interesting behavior and doesn't make the results as stable and consistent as I would have liked.

### Round 2: `GradientBoostingClassifier`

The classifier showed obvious correlation between the parameters selected, accuracy and prediction time. The average accuracy was 79%, with the maximum accuracy of 83% scored at a `max_depth` of 5 and a `num_estimators` of 200. The fastest algorithm was at the other extreme: 50 estimators with a maximum depth of 2. This algorithm took about 0.199 sec per prediction compared with an average of 0.537 sec per prediction.

### Round 3: `SGDClassifier`

The stochastic gradient descent classifier performance has been variable and seems dependent on the seed or other less-understood factors. The `l1` loss function seems most effective. The `hinge` (linear) and `log` loss function seem to exchange the rights for most accurate (paired with `l1`) depending on factors that don't seem to be controlled by the parameters that I'm varying (at least not consistently). Maximum accuracy varies between 77% and 80%. In general, the algorithms all perform the same because the parameters that were varied are training-time parameters. The average prediction time is about 0.018 sec.

## Further Considerations

For robot saftey, the robot should stop immediately if there is a stopsign visible. With a weaker mandate, the robot shouldn't stop working unless there is a stopsign. In terms of metrics, the robot should aim for high recall for saftey and high precision for allowing continued operation.

Further work may be done to tune the two algorithms and estimate their true precision-recall curve. Additional work may also be done to estimate the presence of a stopsign from multiple keypoints instead of a single keypoint. This can be done using machine learning esimations based on the class labels of all of the keypoints or potentially include a filtering/localization technique such as Kalman filtering to estimate the likelihood and relative position of a stopsign.
