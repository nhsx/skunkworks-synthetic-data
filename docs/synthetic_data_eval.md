# Synthetic Data Evaluation

In order to evaluate the privacy, quality and utility of the synthetic data produced, a set of checks were needed. There is not currently an industry standard, so we chose to use an evaluation capability from [Synthetic Data Vault](https://sdv.dev/) (SDV), alongside other approaches which provide a broader range of assessments of the data. SDV’s evaluation capability provides a wide range of metrics which are already implemented, giving a starting point for building a more complete evaluation approach. SDV’s evaluation uses metrics to check whether your synthetic data would be a good substitute for the real data, without causing a change in performance (also known as the utility). The additional checks that were added aimed to make the evaluation of utility more robust, for example by checking there are no identical records in the synthetic and real datasets, but also to provide visual aids to allow the user to see what differences are present in the data. 

## Checks used

The checks both manually implemented and using SDV's evaluation include:
* **Collision analysis** - checking that no two  records are exactly the same in the input and synthetic datasets
* **Correlation analysis** - compares the relationship between the two datasets to see if patterns have been accurately preserved in the synthetic dataset
* **Evaluating the Gower distance** - looking at the closeness of similarity between the input and synthetic datasets to make sure they are not too similar
* **Comparing each dataset using Principal Component Analysis** - reducing the size of the data set to its principal components whilst keeping as much information as possible helps us see how similar the input and synthetic datasets are, and helps us to understand whether the synthetic dataset is useful
* **Propensity testing** - checking whether a model can differentiate between our real and synthetic data.We used a logistic regression model that had been trained on input data. We combined the real and synthetic data then fitted the logistic regression model to the data set. Using the fitted model, we could see how well it differentiated between the real and synthetic data by looking at its ability to predict how likely each row was real or synthetic.
* **Comparison of the Voas-Williamson statistic** - A global goodness of fit metric that compares the variation over degrees of freedom in the synthetic and ground truth data.
* **Comparison of statistical distributions of the features** - to get a high level view of the similarity of the two datasets, the categorical and numerical columns were compared visually. For a more in depth overview of both the real and synthetic datasets we used pandas-profiling to generate reports for each. Pandas profiling is a way of quickly exploring data using just a few lines of code instead of trying to understand every variable.

For more detail on the checks used here, the research papers underlying them as well as more general guidance on ensuring privacy, quality and utility in synthetic data, please see [the thought stream from the NHS Transformation Directorate Analytics Unit](https://nhsx.github.io/AnalyticsUnit/synthetic.html)

## Outputs

The evaluation module produces a HTML report detailing the above checks, with both visuals and metrics for the user to compare.