# Copyright (C) 2013 Wesley Baugh
from nose.tools import assert_raises

from infer import classify, experiment, nlp


class TestExperiment(object):
    test_docs = [
        ('I am happy', 'positive'),
        ('I am sad', 'negative'),
        ('Objective tweets are factual', 'neutral')]
    training_docs = [
        ('Her smile put me in a happy mood', 'positive'),
        ('Why am I so tired? ugh', 'negative'),
        ("I can't wait to see the new movie!", 'positive'),
        ('Not getting an A on a project makes me sad', 'negative'),
        ('I smile if I get a good grade on a project', 'positive')]

    class MyApproach(experiment.Experiment):
        def _setup(self):
            self.nb = classify.MultinomialNB()

        def _get_pickle_objects(self):
            return self.extractor, self.nb

        def _test_data(self):
            for text, label in TestExperiment.test_docs:
                yield self.DataInstance(text=text, label=label)

        def _train_data(self):
            for text, label in TestExperiment.training_docs:
                yield self.DataInstance(text=text, label=label)

        def _train(self, features, label):
            self.nb.train((features, label))

        def _predict(self, features):
            return self.nb.classify(features)

    def setup(self):
        self.extractor = nlp.FeatureExtractor()
        self.extractor.min_n, self.extractor.max_n = 1, 1
        self.experiment = TestExperiment.MyApproach(extractor=self.extractor,
                                                    chunk_size=2, first_chunk=1)

    def test_abstract(self):
        assert_raises(TypeError, experiment.Experiment, (self.extractor, 2, 1))

    def test_iter(self):
        result = iter(self.experiment)

        count, performance = next(result)
        assert count == 1

        count, performance = next(result)
        assert count == 2

        count, performance = next(result)
        assert count == 4

        count, performance = next(result)
        assert count == 5

        assert_raises(StopIteration, next, result)

    def test_iter_flush_no_extra(self):
        self.experiment.chunk_size = 1
        result = iter(self.experiment)

        for i in xrange(1, 6):
            count, performance = next(result)
            assert count == i

        assert_raises(StopIteration, next, result)

    def test_get_misclassified(self):
        self.test_iter()
        result = self.experiment.get_misclassified()
        assert len(result) == 1
        result = result[0]
        assert result.text == TestExperiment.test_docs[-1][0]
        assert result.correct == TestExperiment.test_docs[-1][1]
        assert result.predicted != result.correct
        assert 0 <= float(result.probability) <= 1

    def test_get_pickle_objects(self):
        assert_raises(NotImplementedError, self.experiment.get_pickle_objects)

    def test_repr(self):
        result = repr(self.experiment)
        assert result.startswith('MyApproach')
