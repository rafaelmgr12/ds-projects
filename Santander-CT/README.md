[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rafaelmgr12/ds-projects/blob/main/Santander-CT/Santander-CT.ipynb))

# Description

![https://storage.googleapis.com/kaggle-media/competitions/santander/atm_image.png](https://storage.googleapis.com/kaggle-media/competitions/santander/atm_image.png)

At [Santander](https://www.santanderbank.com/) our mission is to help people and businesses prosper. We are always looking for ways to help our customers understand their financial health and identify which products and services might help them achieve their monetary goals.

Our data science team is continually challenging our machine learning algorithms, working with the global data science community to make sure we can more accurately identify new ways to solve our most common challenge, binary classification problems such as: is a customer satisfied? Will a customer buy this product? Can a customer pay this loan?

In this challenge, we invite Kagglers to help us identify which customers will make a specific transaction in the future, irrespective of the amount of money transacted. The data provided for this competition has the same structure as the real data we have available to solve this problem.

# Evaluation

Submissions are evaluated on [area under the ROC curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic) between the predicted probability and the observed target.

## **Submission File**

For each Id in the test set, you must make a binary prediction of the target variable. The file should contain a header and have the following format:

```
 ID_code,target
 test_0,0
 test_1,1
 test_2,0
 etc.
```
