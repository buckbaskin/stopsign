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

Round 2 primary goals/requirements: 30 fps for predictions (about 0.033 sec per prediction) and accuracy of 69% or higher. This means that if 2 keypoints are classified as stopsigns, then the expectation is that there's an overall image classification accuracy of about 90%. The test data is much larger than 500 keypoint descriptors, so the cutoff for performance at this point is faster than K-Nearest-Neighbors (or at least one set of parameters that outperforms the average KNN).

### `KNeighborsClassifier`

The KNeighborsClassifier was originally chosen for its accuracy. 
It's prediction latency is very lacking. 
It achieves 86% accuracy with 2 neighbors with uniform weights, which is also the fastest algorithm (with little variation based on parameters).
It took about 11.5 sec for it to classify the example data.
This corresponds to about 30 seconds to classify an image, which is about 900x slower than the goal.
When transfered to the test case, the other algorithms should aim for a best case of at least better than 0.1 sec.
The other algorithms may not scale the same way as the `KNeighborsClassifier`, so the true recommendation is that they other algorithms should be orders of magnitude faster.

### `GradientBoostingClassifier`

The classifier showed obvious correlation between the parameters selected, accuracy and prediction time.
The average accuracy was 79%, with the maximum accuracy of 83% scored at a `max_depth` of 5 and a `num_estimators` of 200.
The fastest algorithm was at the other extreme: 50 estimators with a maximum depth of 2.
This algorithm took about 0.199 sec per prediction compared with an average of 0.537 sec per prediction. 

The accuracy meets the requirement for this round, but the prediction latency is likely insufficient for the current goal.

### `SGDClassifier`

The stochastic gradient descent classifier performance has been variable and seems dependent on the seed or other less-understood factors.
The `l1` loss function seems most effective.
The `hinge` (linear) and `log` loss function seem to exchange the rights for most accurate (paired with `l1`) depending on factors that don't seem to be controlled by the parameters that I'm varying (at least not consistently).
Maximum accuracy varies between 77% and 80%.
In general, the algorithms all demonstrate about the same prediction latency because the parameters that were varied are training-time parameters.
The average prediction time is about 0.018 sec.

The accuracy and prediction latency both meet requirements. The latency is a significant improvment over the KNN algorithm with little accuracy penalty (about 6%-8%).

### `MLPClassifier`

The multilayer perceptron classifier had a maximum accuracy of 75% (logistic activation, 1 hidden layer of 200 units).
The average accuracy was 71%. 
The average time to run was 0.645 seconds.
The fastest run time was 0.1378 seconds (relu activation, 1 hidden layer of 50 units). This had about 65% accuracy.
One consideration here is that the algorithm hits a non-convergence warning at 200 iterations.
In practice, this doesn't seem to have significantly impacted the average performance of the neural networks.
On the other hand, the fastest networks aren't as fast as some other algorithms while achieving lower accuracy performance.

The overall algorithm seems to meet the requirements.
It seems that this classifier makes worse latency/accuracy tradeoffs than other classifiers.
Neural network architecture is an open field of investigation, so future work may be done optimizing the tradeoff boundary.

### `SVC`

The support vector classifier appears to be running very slowly.
This appears to largely be training time; however, the prediction time is also on the slower end.
This doesn't bode well for beating the prediction latency of the KNeighborsClassifier.
As of the current writing, the algorithm takes about 3.5 secs to classify and achieves the bare minimum 69% accuracy.
The maximum accuracy was 75% achieved with a polynomial kernel.
The average accuracy was below 40%, and the fastest classification was just over 3 seconds.

The prediction latency appears to not meet the requirements (same order of magnitude as K Nearest Neighbors).
The accuracy may improve on average or with different polynomial parameters but the prediction latency is a deal breaker for now.

Insert obligitory quote about beatings continuing until morale improves.

### `DecisionTreeClassifier`

The decision tree classifer looks promising.
The algorithm averaged 73% accuracy and 0.022 sec prediction latency.
The most accurate algorithm achieved 78% accuracy with the `gini` criterion, a min_split of 2 and a maximum depth of 2.
There are a total of 6 algorithms that achieved the maximum score. All achieved this maximum score with a depth of 2.
There were all possible combinations of the other variables (`gini` v. `entropy`, `min_samples_split`) so it would appear that the only variable with a significant effect was the depth.
This would indicate that the algorithm would potentially overfit in other cases (extra depth).
The fastest classifier was responsive in 0.017 sec on average with the `entropy` criterion, a min_split of 8 and a maximum depth of 2.
This classifier also achieved the maximum accuracy.

The decision tree classifier achieves passing accuracy and very low prediction latency.

## Round 2: Results

The two fastest classifiers were Decision Trees and Stochastic Gradient Descent. 
Both classifiers achieved prediction latency of less than 0.018 sec in multiple configurations.
SGD achieved this performance with 77%, 74%, 73%, 64% accuracy.
Decision Trees achieved this performance with 78% correctly classified (all `max_depth=2` options).

The Decision Tree classifier accuracy was largely invariant to configuration changes.
It may be investigated later.

The SGD classifier accuracy appears more tuneable.
The average accuracy was 66%, but the highest accuracy was more than 10% better. 
The average prediction latency was 0.0183 and the fastest prediction latency was 0.0176, so the prediction latency is largely invariant to the parameter changes.

## Round 2: Optimization 

### SGDClassifier

With multiple different seed prefixes, the SGD is run across the same set of configurations. 
The algorithms that are within 10% of the most accurate algorithm (about 10) are listed.
The top 10 fastest algorithms are listed, although most all of the algorithms predict at essentially the same latency (less than 0.0002 sec when predicting about 500 keypoints at a time). If each list is stable, the analysis will continue with the most promising options (potentially the top 5 or so algorithms that rank well at any seed). If the accuracy list isn't stable, further averaging tests will be run to attempt to find configuration options that generalize better regardless of seed.

After increasing the number of iterations and seeds to 256 per configuration, the SGD performance tuning flatlined and settled to an average of mid to low 60% accuracy.
The horizontal axis is the `max_iter` parameter for the `scikit-learn` [`SGDClassifier` constructor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier).
This indicates that performance gains are dependent on seed and likely don't generalize well.
Based on this data, the decision tree route was investigated.
![Alt Text](https://raw.githubusercontent.com/buckbaskin/stopsign/master/img/sgd_opt/256iters_hinge.png?raw=true "Interesting Alt Text")

### DecisionTreeClassifier

The important seed to vary for the DecisionTreeClassifier is the load_data seed.
This has resulted in quite the variance in low depth classifications.
For both options (`gini` and `entropy`) a depth of about 15 stabilizes the results. 

### Note during optimization

The average recall is very low because many splits of 500 elements have no positive (stopsign) class examples.
This is representative where the image doesn't have a stopsign and is very common in the test data.
The default behavior is to make the metric 0, so the average recall is very low.

## Further Considerations

For robot saftey, the robot should stop immediately if there is a stopsign visible. With a weaker mandate, the robot shouldn't stop working unless there is a stopsign. In terms of metrics, the robot should aim for high recall for saftey and high precision for allowing continued operation.

Further work may be done to tune the two algorithms and estimate their true precision-recall curve. Additional work may also be done to estimate the presence of a stopsign from multiple keypoints instead of a single keypoint. This can be done using machine learning esimations based on the class labels of all of the keypoints or potentially include a filtering/localization technique such as Kalman filtering to estimate the likelihood and relative position of a stopsign.
