infer
=====

A machine learning toolkit for classification and assisted
experimentation.

[![Build Status](https://travis-ci.org/bwbaugh/infer.png?branch=master)](https://travis-ci.org/bwbaugh/infer)

Features
--------

### Assisted experimentation

We provide an experimental foundation for assisted experimentation by
allowing the user to modularize the tasks that are common, but unique in
implementation, to every experiment, including knowing how to:

- Perform any required initial setup to get the model ready.
- Parse the training dataset.
- Parse the testing dataset.
- Train the model given a training instance.
- Make a prediction for a test instance using the current model.

Once the task specific implementations are known, we then coordinate the
use of those functions to perform the experiment in an automated fashion
by automatically performing tasks that can be common to any experiment:

- Run an experiment given an implementation of all the modular tasks.
- Incrementally train the model and get the current model's performance.
- Get the test instances that were incorrectly labeled using the current
    model.

### Feature extraction

A feature extractor takes a document and extracts features to be used
with a machine learning classifier (during training and prediction).

We wanted to provide a way to easily perform the common task of
extracting textual features from documents, while at the same time
making it easy for the researcher to experiment with the kinds of
features that are extracted. The researcher can currently specify:

- The function used to tokenize the raw document string.
- The range of n-grams to extract from the text

### Classification

#### Performance metrics

We provide a means of measuring the performance of your classifier by
providing standard performance metrics, expanded to allow for
multinomial classifiers, including:

- Confusion matrix
- Accuracy
- Recall (average, weighted average, and per class)
- Precision (average, weighted average, and per class)
- F-measure with a selectable *beta* parameter (average, weighted
    average, and per class)

#### (Multinomial) Naive Bayes

Some of the existing implementations of Naive Bayes that are available
in various libraries we have found to be very memory inefficient.
Because of this, we decided to write our own implementation that can
hopefully be better optimized.

In addition, there are lots of ways you can experiment with using and
optimizing the performance of Naive Bayes that we wanted to be able to
easily experiment with.

### Feature selection

Feature selection is another tool that the researcher should be able to
experiment with.

#### Dictionary trimming

Currently, we provide a form of feature selection that is similar to
dictionary trimming, by having the classifier ignore all but the top-k
most frequent features. This often gives us 90% of the benefit of
feature selection without the work of computing more complex metrics.

Dictionary trimming normally involves making a pass over your dataset in
order to extract the (feature) vocabulary. However, this is infeasible
when you are attempting to learn in an online (streaming) setting, such
as when your documents are continuously coming in, like tweets off of a
stream from Twitter. To handle this case, we created a data structure
that *efficiently* keeps track of the top-k most common terms that is
updated while training. This allows O(1) checking if a feature is in the
top-k features *without* having to make a pass over the vocabulary to
make the check.
