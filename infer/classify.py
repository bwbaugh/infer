# Copyright (C) 2013 Wesley Baugh
"""Tools for text classification."""
from __future__ import division

import abc
import math
from collections import defaultdict, namedtuple, Counter
from fractions import Fraction

import nltk


class MostCommon(object):
    """Keep track the top-k key-value pairs.

    Attributes:
        top: Integer representing the top-k items to keep track of.
        store: Dictionary of the top-k items.
        min: The current minimum of any top-k item.
        min_set: Set where keys are counts, and values are the set of
            keys with that count.
    """
    def __init__(self, top):
        """Create a new MostCommon object to track key-value paris.

        Args:
            top: Integer representing the top-k values to keep track of.
        """
        self.top = top
        self.store = dict()
        self.min = None
        self.min_set = defaultdict(set)

    def _update_existing(self, key, value):
        """Update an item that is already one of the top-k values."""
        # Currently handle values that are non-decreasing.
        assert value > self.store[key]
        self.min_set[self.store[key]].remove(key)
        if self.store[key] == self.min:  # Previously was the minimum.
            if not self.min_set[self.store[key]]:  # No more minimums.
                del self.min_set[self.store[key]]
                self.min_set[value].add(key)
                self.min = min(self.min_set.keys())
        self.min_set[value].add(key)
        self.store[key] = value

    def __contains__(self, key):
        """Boolean if the key is one of the top-k items."""
        return key in self.store

    def __setitem__(self, key, value):
        """Assign a value to a key.

        The item won't be stored if it is less than the minimum (and
        the store is already full). If the item is already in the store,
        the value will be updated along with the `min` if necessary.
        """
        # Store it if we aren't full yet.
        if len(self.store) < self.top:
            if key in self.store:  # We already have this item.
                self._update_existing(key, value)
            else:  # Brand new item.
                self.store[key] = value
                self.min_set[value].add(key)
                if value < self.min or self.min is None:
                    self.min = value
        else:  # We're full. The value must be greater minimum to be added.
            if value > self.min:  # New item must be larger than current min.
                if key in self.store:  # We already have this item.
                    self._update_existing(key, value)
                else:  # Brand new item.
                    # Make room by removing one of the current minimums.
                    old = self.min_set[self.min].pop()
                    del self.store[old]
                    # Delete the set if there are no old minimums left.
                    if not self.min_set[self.min]:
                        del self.min_set[self.min]
                    # Add the new item.
                    self.min_set[value].add(key)
                    self.store[key] = value
                    self.min = min(self.min_set.keys())

    def __repr__(self):
        if len(self.store) < 10:
            store = repr(self.store)
        else:
            length = len(self.store)
            largest = max(self.store.itervalues())
            store = '<len={length}, max={largest}>'.format(length=length,
                                                           largest=largest)
        return ('{self.__class__.__name__}(top={self.top}, min={self.min}, '
                'store={store})'.format(self=self, store=store))


class Classifier(object):
    """Abstract base class for classifiers."""
    __metaclass__ = abc.ABCMeta
    Prediction = namedtuple('Prediction', 'label confidence')


class MultinomialNB(Classifier):
    """Multinomial Naive Bayes for text classification.

    Attributes:
        exact: Boolean indicating if exact probabilities should be
            returned as a `Fraction`. Otherwise, speed up computations
            but only return probabilities as a `float`. (default False)
        laplace: Smoothing parameter >= 0. (default 1)
        top_features: Number indicating the top-k most common features
            to use during classification, sorted by the frequency the
            feature has been seen (a count is kept for each label). This
            is a form of feature selection because any feature that has
            a frequency less than any of the top-k most common features
            is ignored during classification. This value must be set
            before any training of the classifier. (default None)

    Properties:
        labels: Set of all class labels.
        vocabulary: Set of vocabulary across all class labels.
    """
    def __init__(self, *documents):
        """Create a new Multinomial Naive Bayes classifier.
        Args:
            documents: Optional list of document-label pairs for training.
        """
        self.exact = False
        self.laplace = 1
        self.top_features = None
        # Dictionary of sets of vocabulary by label.
        self._label_vocab = defaultdict(set)
        # Dictionary of times a label has been seen.
        self._label_count = Counter()
        # Dictionary of number of feature seen in all documents by label.
        self._label_length = Counter()
        # Dictionary of times a feature has been seen by label.
        self._label_feature_count = defaultdict(Counter)
        # Size of vocabulary across all class labels.
        self._vocab_size = 0
        if documents:
            self.train(*documents)

    @property
    def labels(self):
        """Set of all class labels.

        Returns:
            Example: set(['positive', 'negative'])
        """
        return set(label for label in self._label_count)

    @property
    def vocabulary(self):
        """Set of vocabulary (features) seen in any class label."""
        label_vocab = [self._label_vocab[x] for x in self._label_vocab]
        return set().union(*label_vocab)

    def train(self, *documents):
        """Train the classifier on a document-label pair(s).

        Args:
            documents: Tuple of (document, label) pair(s). Documents
                must be a collection of features. The label can be any
                hashable object, though is usually a string.
        """
        for document, label in documents:
            # Python 3: isinstance(document, str)
            if isinstance(document, basestring):
                raise TypeError('Documents must be a collection of features')
            self._label_count[label] += 1
            for feature in document:
                # Check if the feature hasn't been seen before for any label.
                if not any(feature in self._label_vocab[x] for x in self.labels):
                    self._vocab_size += 1
                self._label_vocab[label].add(feature)
                self._label_feature_count[label][feature] += 1
                self._label_length[label] += 1
                if self.top_features:
                    if not hasattr(self, '_most_common'):
                        x = lambda: MostCommon(self.top_features)
                        self._most_common = defaultdict(x)
                    y = self._label_feature_count[label][feature]
                    self._most_common[label][feature] = y

    def prior(self, label):
        """Prior probability of a label.

        Args:
            label: The target class label.
            self.exact

        Returns:
            The number of training instances that had the target
            `label`, divided by the total number of training instances.
        """
        if label not in self.labels:
            raise KeyError(label)
        total = sum(self._label_count.values())
        if self.exact:
            return Fraction(self._label_count[label], total)
        else:
            return self._label_count[label] / total

    def conditional(self, feature, label):
        """Conditional probability for a feature given a label.

        Args:
            feature: The target feature.
            label: The target class label.
            self.laplace
            self.exact

        Returns:
            The number of times the feature has been present across all
            training documents for the `label`, divided by the sum of
            the length of every training document for the `label`.
        """
        # Note we use [Laplace smoothing][laplace].
        # [laplace]: https://en.wikipedia.org/wiki/Additive_smoothing
        if label not in self.labels:
            raise KeyError(label)

        # Times feature seen across all documents in a label.
        numer = self.laplace
        # Avoid creating an entry if the term has never been seen
        if feature in self._label_feature_count[label]:
            numer += self._label_feature_count[label][feature]
        denom = self._label_length[label] + (self._vocab_size * self.laplace)
        if self.exact:
            return Fraction(numer, denom)
        else:
            return numer / denom

    def _score(self, document, label):
        """Multinomial raw score of a document given a label.

        Args:
            document: Collection of features.
            label: The target class label.
            self.exact

        Returns:
            The multinomial raw score of the `document` given the
            `label`. In order to turn the raw score into a confidence
            value, this value should be divided by the sum of the raw
            scores across all class labels.
        """
        if isinstance(document, basestring):
            raise TypeError('Documents must be a list of features')

        if self.exact:
            score = self.prior(label)
        else:
            score = math.log(self.prior(label))

        for feature in document:
            # Feature selection by only considering the top-k
            # most common features (a form of dictionary trimming).
            if self.top_features and feature not in self._most_common[label]:
                continue
            conditional = self.conditional(feature, label)
            if self.exact:
                score *= conditional
            else:
                score += math.log(conditional)

        return score

    def _compute_scores(self, document):
        """Compute the multinomial score of a document for all labels.

        Args:
            document: Collection of features.

        Returns:
            A dict mapping class labels to the multinomial raw score
            for the `document` given the label.
        """
        return {x: self._score(document, x) for x in self.labels}

    def prob_all(self, document):
        """Probability of a document for all labels.

        Args:
            document: Collection of features.
            self.exact

        Returns:
            A dict mapping class labels to the confidence value that the
            `document` belongs to the label.
        """
        score = self._compute_scores(document)
        if not self.exact:
            # If the log-likelihood is too small, when we convert back
            # using `math.exp`, the result will round to zero.
            normalize = max(score.itervalues())
            assert normalize <= 0, normalize
            score = {x: math.exp(score[x] - normalize) for x in score}
        total = sum(score[x] for x in score)
        assert total > 0, (total, score, normalize)
        if self.exact:
            return {label: Fraction(score[label], total) for label in
                    self.labels}
        else:
            return {label: score[label] / total for label in self.labels}

    def prob(self, document, label):
        """Probability of a document given a label.

        Args:
            document: Collection of features.
            label: The target class label.

        Returns:
            The confidence value that the `document` belongs to `label`.
        """
        prob = self.prob_all(document)[label]
        return prob

    def classify(self, document):
        """Get the most confident class label for a document.

        Args:
            document: Collection of features.

        Returns:
            A namedtuple representing the most confident class `label`
            and the value of the `confidence` in the label. For example:

            As tuple:
                ('positive', 0.85)
            As namedtuple:
                Prediction(label='positive', confidence=0.85)
        """
        prob = self.prob_all(document)
        label = max(prob, key=prob.get)
        return self.Prediction(label, prob[label])


def evaluate(reference, test, beta=1):
    """Compute various performance metrics.

    Args:
        reference: An ordered list of correct class labels.
        test: A corresponding ordered list of class labels to evaluate.
        beta: A float parameter for F-measure (default = 1).

    Returns:
        A dictionary with an entry for each metric.
    """
    performance = dict()

    # We can compute nearly everything from a confusion matrix.
    matrix = nltk.confusionmatrix.ConfusionMatrix(reference, test)
    performance['confusionmatrix'] = matrix

    # Number of unique labels; used for computing averages.
    num_labels = len(matrix._confusion)

    # Accuracy
    performance['accuracy'] = matrix._correct / matrix._total

    # Recall
    # correctly classified positives / total positives
    average = weighted_average = 0
    for label, index in matrix._indices.iteritems():
        true_positive = matrix._confusion[index][index]
        total_positives = sum(matrix._confusion[index])
        if total_positives == 0:
            recall = int(true_positive == 0)
        else:
            recall = true_positive / total_positives
        average += recall
        weighted_average += recall * total_positives
        key = 'recall-{0}'.format(label)
        performance[key] = recall
    performance['average recall'] = average / num_labels
    performance['weighted recall'] = weighted_average / matrix._total

    # Precision
    # correctly classified positives / total predicted as positive
    average = weighted_average = 0
    for label, index in matrix._indices.iteritems():
        true_positive = matrix._confusion[index][index]
        total_positives = sum(matrix._confusion[index])
        predicted_positive = 0  # Subtract true_positive to get false_positive
        for i in xrange(num_labels):
            predicted_positive += matrix._confusion[i][index]
        if true_positive == predicted_positive == 0:
            precision = 1
        else:
            precision = true_positive / predicted_positive
        average += precision
        weighted_average += precision * total_positives
        key = 'precision-{0}'.format(label)
        performance[key] = precision
    performance['average precision'] = average / num_labels
    performance['weighted precision'] = weighted_average / matrix._total

    # F-Measure
    # ((1 + B ** 2) * precision * recall) / (((B ** 2) * precision) + recall)
    average = weighted_average = 0
    for label, index in matrix._indices.iteritems():
        recall = performance['recall-{0}'.format(label)]
        precision = performance['precision-{0}'.format(label)]
        total_positives = sum(matrix._confusion[index])
        numer = ((1 + beta ** 2) * precision * recall)
        denom = (((beta ** 2) * precision) + recall)
        if denom > 0:
            f_measure = numer / denom
        else:
            f_measure = 0
        average += f_measure
        weighted_average += f_measure * total_positives
        key = 'f-{0}'.format(label)
        performance[key] = f_measure
    performance['average f_measure'] = average / num_labels
    performance['weighted f_measure'] = weighted_average / matrix._total

    return performance
