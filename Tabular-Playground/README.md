# Tabular Playground Series - Sep 2021

**Practice your ML skills on this approachable dataset!**

## Description

Kaggle competitions are incredibly fun and rewarding, but they can also be intimidating for people who are relatively new in their data science journey. In the past, we've launched many Playground competitions that are more approachable than our Featured competitions and thus, more beginner-friendly.

In order to have a more consistent offering of these competitions for our community, we're trying a new experiment in 2021. We'll be launching month-long tabular Playground competitions on the 1st of every month and continue the experiment as long as there's sufficient interest and participation.

The goal of these competitions is to provide a fun, and approachable for anyone, tabular dataset. These competitions will be great for people looking for something in between the Titanic Getting Started competition and a Featured competition. If you're an established competitions master or grandmaster, these probably won't be much of a challenge for you. We encourage you to avoid saturating the leaderboard.

For each monthly competition, we'll be offering Kaggle Merchandise for the top three teams. And finally, because we want these competitions to be more about learning, we're limiting team sizes to 3 individuals.

The dataset is used for this competition is synthetic, but based on a real dataset and generated using a [CTGAN](https://github.com/sdv-dev/CTGAN). The original dataset deals with predicting whether a claim will be made on an insurance policy. Although the features are anonymized, they have properties relating to real-world features.

Good luck and have fun!

For ideas on how to improve your score, check out the [Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) and [Intermediate Machine Learning](https://www.kaggle.com/learn/intermediate-machine-learning) courses on Kaggle Learn.

## Evaluation

Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

### **Submission File**

For each `id` in the test set, you must predict a probability for the `claim` variable. The file should contain a header and have the following format:

```
id,claim
957919,0.5
957920,0.5
957921,0.5
etc.
```