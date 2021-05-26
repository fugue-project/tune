Top Level API Reference
=======================

.. |SpaceTutorial| replace:: :ref:`Space Tutorial </notebooks/space.ipynb>`
.. |Trial| replace:: :class:`~tune.concepts.flow.trial.Trial`
.. |SortMetric| replace:: :meth:`tune.concepts.flow.report.TrialReport.sort_metric`
.. |TrialObject| replace:: :class:`~tune.concepts.flow.trial.Trial`
.. |TrialReportObject| replace:: :class:`~tune.concepts.flow.report.TrialReport`
.. |NonIterativeObjective| replace:: a :class:`~tune.noniterative.objective.NonIterativeObjectiveFunc` compatible object, please read :ref:`Non-Iterative Objective Explained </notebooks/noniterative_objective.ipynb>`

.. contents::
   :local:

Space Concept
-------------

Space
~~~~~

.. autoclass:: tune.concepts.space.spaces.Space
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Grid
~~~~

.. autoclass:: tune.concepts.space.parameters.Grid
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Choice
~~~~~~

.. autoclass:: tune.concepts.space.parameters.Choice
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Rand
~~~~

.. autoclass:: tune.concepts.space.parameters.Rand
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

RandInt
~~~~~~~

.. autoclass:: tune.concepts.space.parameters.RandInt
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:


General Non-Iterative Problems
------------------------------

.. autofunction:: tune.api.suggest.suggest_for_noniterative_objective
   :noindex:

.. autofunction:: tune.api.optimize.optimize_noniterative
   :noindex:


General Iterative Problems
--------------------------

Successive Halving
~~~~~~~~~~~~~~~~~~

.. autofunction:: tune.api.suggest.suggest_by_sha
   :noindex:

.. autofunction:: tune.api.optimize.optimize_by_sha
   :noindex:

Hyperband
~~~~~~~~~

.. autofunction:: tune.api.suggest.suggest_by_hyperband
   :noindex:

.. autofunction:: tune.api.optimize.optimize_by_hyperband
   :noindex:

Continuous ASHA
~~~~~~~~~~~~~~~

.. autofunction:: tune.api.suggest.suggest_by_continuous_asha
   :noindex:

.. autofunction:: tune.api.optimize.optimize_by_continuous_asha
   :noindex:


For Scikit-Learn
----------------

.. autofunction:: tune_sklearn.utils.sk_space
   :noindex:

.. autofunction:: tune_sklearn.suggest.suggest_sk_models_by_cv
   :noindex:

.. autofunction:: tune_sklearn.suggest.suggest_sk_models
   :noindex:



For Tensorflow Keras
--------------------

.. autoclass:: tune_tensorflow.spec.KerasTrainingSpec
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autofunction:: tune_tensorflow.utils.keras_space
   :noindex:

.. autofunction:: tune_tensorflow.suggest.suggest_keras_models_by_continuous_asha
   :noindex:

.. autofunction:: tune_tensorflow.suggest.suggest_keras_models_by_hyperband
   :noindex:

.. autofunction:: tune_tensorflow.suggest.suggest_keras_models_by_sha
   :noindex:
