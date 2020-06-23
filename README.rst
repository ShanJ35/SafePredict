.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/SafePredict/badge/?version=latest
.. _ReadTheDocs: https://SafePredict.readthedocs.io/en/latest/?badge=latest

SafePredict 
============================================================

SafePredict is an online learning meta-algorithm that uses refusals to guarantee asymptotic error bounds regardless of underlying predictor or data distribution.

Further details about parameters and the SafePredict algorithm can be found in the original paper here: https://arxiv.org/abs/1708.06425

Installation
------------

Dependencies
~~~~~~~~~~~~

Use ``pip`` to install required dependencies ::

    pip install threadpoolctl joblib numpy scikit-learn

SafePredict
~~~~~~~~~~~~

Use ``pip`` to install SafePredict ::

    pip install -i https://test.pypi.org/simple/ SafePredict

Usage
------------

SafePredict has one primary object called ``safePredict``. It implements the primary methods as other scikit-learn estimators, namely ``fit, partial_fit, predict``
and is compatible with scikit-learn estimators and pipelines. 

``fit(X, y, random_state)`` calls the ``fit`` method of the base predictor, or the ``partial_fit`` method if available, and updates the parameters of SafePredict using each datapoint in ``(X,y)`` sequentially. 
``partial_fit(X, y, random_state)`` has the same functionality as ``fit(X, y, random_state)`` due to the online-learning setting. 

``predict(X, random_state)`` predicts the labels according to the base predictor and whether SafePredict accepts the base prediction or not. There are no updates made to the base predictor or SafePredict in this method. 


The following are parameters for the ``safePredict`` object ::

    base_estimator: Base predictor to be used, must be a valid scikit-learn estimator. Default: sklearn.ensemble.RandomForestClassifier()

    base_estimator_params: Tuple containing parameters for the base estimator. Default: ()

    target_error: Float describing asymptotic error rate to be achieved by SafePredict. Default: 0.1

    dummy_prob: Float describing initial probability of SafePredict refusing a base prediction. Default: 0.5

    prediction_prob: Float describing initial probability of SafePredict accepting a base prediction. 
    Note: dummy_prob and prediction_prob must sum to 1.0 . Default: 0.5

    alpha: Float describing adaptivity parameter which is strict lower bound on the probability that SafePredict accepts a base prediction. 
    Note: Validity is guaranteed if alpha is on the order of 1/T where T is the horizon. Default: 0.0

    beta: Float describing adaptivity parameter which is strict upper bound on the probability that SafePredict accepts a base prediction. 
    Note: Efficiency is optimal if beta is 2. Default: 1.0

    horizon: Integer describing number of data points to be trained on in the online-learning setting. Default: 1

    refusal_class: Integer describing what label SafePredict should output if it refuses a base prediction. Default: -1

    calibration: String describing which calibration method, if any, to use for confidence based refusals combined with SafePredict. 
    Supports "sigmoid", "isotonic", None. Default: None

    random_state: Integer for reproducible outputs across multiple function calls. Default: None


