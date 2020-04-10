import pytest

from sklearn.utils.estimator_checks import check_estimator

from SafePredict import TemplateEstimator
from SafePredict import TemplateClassifier
from SafePredict import TemplateTransformer


@pytest.mark.parametrize(
    "Estimator", [TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
