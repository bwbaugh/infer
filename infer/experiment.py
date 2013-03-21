# Copyright (C) 2013 Wesley Baugh
"""Tools for assisting experimentation."""
import abc
from collections import namedtuple

from infertweet import classify


class Experiment(object):
    """Provides a programmatic way of assisting experimentation.

    For any given machine learning task, we often perform experiments by
    setting the training and test data, along with trying out various
    combinations of feature extraction strategies and tuning variable
    values. The usual way this is done is to:

    1. Write a script to perform the experiment.
    2. Run the experiment and get the performance.
    3. Make changes either in a new copy of the script or by commenting
        out the old code in-place
    4. Get the new performance and repeat the process.

    We provide an experimental foundation for assisted experimentation
    by allowing the user to modularize the tasks that are common but
    unique in implementation to every experiment, including knowing how
    to:

    - Perform any required initial setup to get the model ready.
    - Parse the training dataset.
    - Parse the testing dataset.
    - Train the model given a training instance.
    - Make a prediction for a test instance using the current model.

    Once the task specific implementations are known, we then coordinate
    the use of those functions to perfrom the experiment in an automated
    fashion by automatically performing tasks that can be common to any
    experiment:

    - Run an experiment given an implementation of all the modular
        tasks.
    - Incrementally train the model and get the current model's
        performance.
    - Get the test instances that were incorrectly labeled using the
        current model.

    To use, implement all of the abstract methods, then simply iterate
    over the concrete instance to get the performance of the model
    periodically during training.

    Note that you create multiple subclasses that each implement only a
    single abstract method. Each of those subclasses act as a mixin,
    which can then all be combined together into a single concrete
    implementation. This is useful when you wish to test the same model
    using different training or test sets, because you can re-use common
    code and mixin any that is unique to the current approach.

    Class Variables:
        DataInstance: A namedtuple with `text` and `label` attributes,
            usually yielded when accessing the training and test data.
        Misclassified: A namedtuple representing a test instance that
            was misclassified using the current model.
                Attributes:
                    probability: The confidence in the predicted label.
                    correct: The correct label for the test instance.
                    predicted: The (incorrect) label that was predicted.
                    text: The text of the test instance.

    Attributes:
        extractor: An instance of nlp.FeatureExtractor used to extract
            features from the text of the training and test instances,
            which are then used for training or classifying.
        chunk_size: Number of instances to use for training before next
            yielding the current performance on the test data.
        first_chunk: The current performance will also be yielded after
            this number of training instances. Useful for getting the
            results after the performance has stabilized (such as after
            50 - 100 training instances).
    """
    __metaclass__ = abc.ABCMeta
    DataInstance = namedtuple('DataInstance', 'text label')
    Misclassified = namedtuple('Misclassified', 'probability correct '
                                                'predicted text')

    def __init__(self, extractor, chunk_size, first_chunk=0, test_scale=1,
                 evaluator=classify.evaluate):
        """Create an Experiment instance.

        Args:
            extractor: An instance of nlp.FeatureExtractor used to
                extract features from the text of the training and test
                instances, which are then used for training or
                classifying.
            chunk_size: Number of instances to use for training before
                next yielding the current performance on the test data.
            first_chunk: The current performance will also be yielded
                after this number of training instances. Useful for
                getting the results after the performance has stabilized
                (such as after 50 - 100 training instances). (default 0)
            test_scale: Only use the first (1 / test_scale) instances
                of the test set to calculate performance. Useful during
                the development process if testing on the full test set
                takes a relatively long time if the chunk_size is small.
                (default 1 (no scaling))
            evaluator: A function that evaluates the performance of the
                model by comparing the correct labels of the test set
                to the labels predicted by the current model.
                (default classify.evaluate)
        """
        self.extractor = extractor
        self.chunk_size = chunk_size
        self.first_chunk = first_chunk
        self.test_scale = test_scale
        self._evaluate = evaluator
        self._correct_labels, self._test_data = self._pack_test_data()
        self._setup()

    # An optional abstractmethod.
    def _setup(self):
        """Create the model (classifiers) or any other initial setup."""
        pass  # pragma: no cover

    # An optional abstractmethod
    def get_pickle_objects(self):
        """Get objects that are necessary to recreate the model."""
        # Usually `self.extractor` and any objects (such as the
        # classifier) created in `self._setup`.
        raise NotImplementedError

    @abc.abstractmethod
    def _test_data():  # @staticmethod
        """Parse the test dataset and get test instances.

        Yields:
            A namedtuple with the document `text` and class `label` for
            each instance in the test set. For example:

            TestInstance(text='...', label='positive')
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def _train_data(self):
        """Parse the training dataset and get training instances.

        Yields:
            A namedtuple with the document `text` and the correct class
            `label` of a single training document. For example:

            As as tuple:
                text, label = ('...', 'positive')
            As as namedtuple:
                DataInstance(text='...', label='positive')
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def _train(self, features, label):
        """Update the training model given features and a label.

        Args:
            features: Collection of extracted features.
            label: The correct class label.
        """
        pass  # pragma: no cover

    @abc.abstractmethod
    def _predict(self, features):
        """Predict the class label given the extracted features.

        Args:
            features: Collection of extracted features.

        Returns:
            A namedtuple with the predicted class `label` and the
            `confidence` of the prediction returned by the classifier.
            For example:

            As tuple:
                ('positive', 0.85)
            As namedtuple:
                Prediction(label='positive', confidence=0.85)
        """
        pass  # pragma: no cover

    def _pack_test_data(self):
        """Packs the test data into lists for storing in memory.

        Returns:
            A tuple containing an ordered list of `correct_labels`, and
            an ordered list of namedtuple objects that contain the
            `text` and correct class `label` of the test data. For
            example:

            (['positive', 'positive', 'negative'],
             [DataInstance(text='...', label='positive'),
              DataInstance(text='...', label='positive')
              DataInstance(text='...', label='negative')])
        """
        correct_labels, test_data = [], []
        for instance in self._test_data():
            correct_labels.append(instance.label)
            test_data.append(instance)
        return (correct_labels[:len(correct_labels) / self.test_scale],
                test_data[:len(test_data) / self.test_scale])

    def _test_performance(self):
        """Run the classifier on the test data and calculate performance."""
        predicted_labels = []
        for instance in self._test_data:
            document = instance.text
            features = self.extractor.extract(document)
            label, probability = self._predict(features)
            predicted_labels.append(label)
        performance = self._evaluate(self._correct_labels, predicted_labels)
        return performance

    def get_misclassified(self):
        """Get currently misclassified instances of the test set."""
        misclassified = []
        for instance in self._test_data:
            document = instance.text
            features = self.extractor.extract(document)
            label, probability = self._predict(features)
            if label != instance.label:
                x = self.Misclassified(probability=probability,
                                       correct=instance.label,
                                       predicted=label,
                                       text=instance.text)
                misclassified.append(x)
        return misclassified

    def __iter__(self):
        """Get the performance on a test set during training.

        Yields:
            Tuple with the current integer `count` of the number of
            instances the classifier has been trained on, and a
            `performance` dict mapping each performance metric to the
            current value as evaluated on the test set.
        """
        for count, instance in enumerate(self._train_data(), start=1):
            features = self.extractor.extract(instance.text)
            self._train(features, instance.label)
            if count % self.chunk_size == 0 or count == self.first_chunk:
                yield count, self._test_performance()
        # Flush if necessary.
        if count % self.chunk_size != 0:
            yield count, self._test_performance()

    def __repr__(self):
        message = '{0}(extractor={self.extractor!r})'
        return message.format(self.__class__.__name__, self=self)
