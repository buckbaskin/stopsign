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

## Round 2: Results Part 1

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
For both options (`gini` and `entropy`) a depth of about 15 stabilizes the results at about 70% accuracy.
Moving forward, the default option (`gini`) will be used with a max_depth of 15.
![Alt Text](https://raw.githubusercontent.com/buckbaskin/stopsign/master/img/dtc_opt/gini_008_avg.png?raw=true "Interesting Alt Text")

### Note during optimization

The average recall is very low because many splits of 500 elements have no positive (stopsign) class examples.
This is representative where the image doesn't have a stopsign and is very common in the test data.
The default behavior is to make the metric 0, so the average recall is very low.

## Round 2: Results Part 2

The resulting classification can be visualized with the demo video. [Demo!](https://youtu.be/ZV9dfaegjz0)
The classifier worked well when presented with complete data from all of the video.
When the classifier was applied to new data, it did not properly classify stopsigns.
Further attempts were made to classify the first video with new data from an expanded multi-video set collected in a variety of lighting situations, rotations and viewing angles.
This new dataset is complex enough that it did not show good performance with the fast classifiers, either demonstrating excessive positive prediction or virtually no positive prediction and poor (30%) recall.
Recall was identified as a key metric for vision to be used as a saftey system and the current algorithms demonstrate insufficient recall to be at all useful for this task.
Collaborative classification by grouping potential positive results proved unhelpful because the algorithm rarely predicted positive results in the region near the stopsign.

## Round 3: Moving Forward

Goals:

    - Increase recall
    - Use multi-video data collection to create a robust classifier
    - Explore dimensionality reduction to improve speed of slower algorithms and/or increase score performance for existing fast algorithms
    - Gain a better understanding of postive (stopsign) keypoints. This includes quantifying how many there are in the current dataset, how many are typically present on a manually identified stopsign, and any visible trends in the values of the descriptor.
    - Explore adding additional features:
    	- pixel color
    	- average region color
    	- RGB values for above
    	- HSV values for above
    	- histogram of colors in region (ex. number of pixels that are red, green, blue, yellow, orange, white, black. A stopsign should have red and possibly white.

Going back to basic principles for exploring the data, 2D plots of different combinations of values for descriptors and potential new features will be visualized and additional correlation measures will be introduced.
While blind attempts at optimization without understanding any patterns in the data was a fun exercise, new datasets have proven intractable to blind search.

All algorithms are back on the table to be tested with dimensionality reduction on sets of 500 elements (a typical number of keypoints in an image).
The combined dimensionality reduction and prediction will be required to meet the reduced goal of 10 fps average.

Additional measures will be taken to reduce the impact of datasets with no positive examples.
These lead to recall being set to 0 and dilute potentially high-recall classifiers.
Additional measures may be taken to reduce the impact of datasets with no predicted positive examples.
This may dilute the precision values seen for the classifier, but time, accuracy and recall are higher priorities.

### Visualization

See Visual_Exploration.

### Promising Algorithms

Using some basic tips from a web page I found, I started tracking accuracy for both the training and test datasets.
If the accuracy is low for both, the classifier has High Bias.
If the accuracy is lower for the test data set, the model is overfitting and has High Variance.
The other potential tradeoff for the algorithm is high precision vs. high recall.

In the process of testing some new algorithms on bitwise data (instead of aggregated integer data) the AdaBoosted Decision Tree Regressor/Classifier shows promise.
Its performance degrades with increasing maximum depth; however, initial test runs with a large number of boosting steps reached nearly 100% classification of training data and increasing accuracy on the test data.
It also showed increasing precision and recall. These may be split into test and training evaluations as well in the future to aid in identification of bias or variance.

The model above can be tuned using decision tree parameters (primarily `max_depth`), AdaBoost parameters (`n_estimators`, `learning_rate`, `loss`) and the thresholding value (currently 0.5).
The thresholding value will be explored as a tradeoff between precision and recall at any parameter set.
The parameters will be tuned to maximize accuracy while maintaining the 30fps prediction rate (< 0.033 sec per prediction).

### Exploring Boosted Decision Trees

The model accuracy and other parameters increase with depth (both test and training); however, there is an observed significant jump in prediction latency with a maximum depth of 7 or more that violates the prediction latency requirement.
This could be mitigated by changing the learning rate the achieve similar results with fewer estimators with a `max_depth` of 7 or more.
This could also be mitigated by running two predictors in parallel.
This does not fix high latency, but it would potentially allow for increased throughput to predict for 30 frames every second, with two predictors of not less than 0.067 sec prediction latency.
Each predictor would predict either even or odd frames.
The robot would then travel at most 13 cm at maximum speed before the prediction would occur and likely be able to stop within 20 cm and less than 1 foot.

Of particular note during this testing, the model maximum depth had a significant effect on model latency. A depth of 6 or lower met latency requirements and a depth of 7 or greater did not when run with 300 estimators in the boosting.
The model with a depth of 7 could be found to meet requirements with fewer estimators; however, the number of estimators did not significantly affect the performance (accuracy or latency) of a model of depth 6.

| Max Depth | N Estimators  | Pred Latency  | Train acc | Test acc  |
| ---       | ---           | ---           | ---       | ---       |
| 20        | 91            | 0.026 sec     | 100%      | 75%       |
| 20        | 121           | ~~0.034 sec~~ | 100%      | 76%       |
| 10        | 91            | 0.026 sec     | 100%      | 78%       |
| 10        | 121           | ~~0.034 sec~~ | 100%      | 78%       |
| 9         | 121           | 0.031 sec     | 100%      | 78%       |
| 9         | 151           | ~~0.039 sec~~ | 100%      | 78%       |
| 9         | 300           | ~~0.079 sec~~ | 100%      | 78%       |
| 8         | 121           | 0.029 sec     | 99%       | 77%       |
| 8         | 151           | ~~0.036 sec~~ | 99%       | 77%       |
| 7         | 121           | 0.027 sec     | 94%       | 76%       |
| 7         | 151           | ~~0.034 sec~~ | 95%       | 76%       |
| 7         | 300           | ~~0.068 sec~~ | 99%       | 77%       |
| 6         | 31            | 0.003 sec     | 72%       | ~~67%~~   |
| 6         | 300           | 0.003 sec     | 73%       | ~~68%~~   |
| 5         | 300           | <0.001 sec    | ~~63%~~   | ~~61%~~   |

The data suggests that one can achieve maximal training accuracy with a `max_depth` of 9.
This means that decision tree models won't gain any training benefit by getting deeper with the current dataset.
The data presented here suggests that the model begins to excessively overfit somewhere greater than a depth of 10 (decreasing test accuracy for the same train accuracy); however the maximum depth may not be reached because the prediction latency of the models is effectively the same.
The maximum performance appears to be attainable with a `n_estimators` value of 121.
Work was not done to push this value down. It is considered comfortably within the prediction latency requirement.
On the other hand, reducing this parameter may help avoid overfitting. 

Precision and recall are largely balanced at this point in the evaluation. No tuning of the threshold has been done up to this point.
The model does appear to have some overfitting (high variance).
The recommended changes are to reduce features or increase the size of the dataset.

## Round 3: Video Attempt 1

The labelling for the first dataset shows that the model's test accuracy isn't reliable enough for use in the actual video data. For example, consider two pictures. These come from the 75th and 76th frames of the video. The image (upon visual inspection) is similar, but one contains multiple stopsign identifications on the stopsign (75, large green circles) and one does not (76). The model ran with essentially 99% training accuracy and 78% test accuracy, so this data proves that the model is overfitting the training data.

### 75

![Alt Text](https://raw.githubusercontent.com/buckbaskin/stopsign/master/img/bdt_v2/frame0075.jpg?raw=true "Interesting Alt Text")

### 76

![Alt Text](https://raw.githubusercontent.com/buckbaskin/stopsign/master/img/bdt_v2/frame0076.jpg?raw=true "Interesting Alt Text")

There are two avenues that will leverage existing code that will be attempted before moving to additional code.
First, the video will be generated with a high recall threshold classifier.
This may fix the issue, but the second step will likely also be useful. 
I'm going to label an additional 500 random images.
This will become an expanded training dataset, which will first be used like the 200 image dataset.
After that, the 200 image dataset will be used as verification for the accuracy of the classifier trained on the complete 500 image dataset to better estimate the generalization of the classifier.

Bagging and other classifiers combined with boosting/bagging will be investigated if the current algorithm doesn't generalize well with the existing data.

#### Future Steps

    5. Evaluate model for bias, variance, precision, recall tradeoffs.
        - High Bias: add additional model features, additional training data
        - High Variance: add training examples, tune model parameters for maximization on the training set
        - Low Precision: tune threshold
        - Low Recall: tune threshold
    4. Pick a set of parameters. Plot precision/recall curves for those parameters. Measure the area under the curve.
    6. Exhaustively compare all models that meet the prediction latency requirement. Identify those models with higher areas under the curve (Area under ROC)
    7. Pick a model and tune the precision/recall on actual video data

## Further Considerations

For robot saftey, the robot should stop immediately if there is a stopsign visible. With a weaker mandate, the robot shouldn't stop working unless there is a stopsign. In terms of metrics, the robot should aim for high recall for saftey and high precision for allowing continued operation.

Further work may be done to tune the two algorithms and estimate their true precision-recall curve. Additional work may also be done to estimate the presence of a stopsign from multiple keypoints instead of a single keypoint. This can be done using machine learning esimations based on the class labels of all of the keypoints or potentially include a filtering/localization technique such as Kalman filtering to estimate the likelihood and relative position of a stopsign.
