Top Level API Reference
=======================

.. |SpaceTutorial| replace:: :ref:`Space Tutorial </notebooks/space.ipynb>`
.. |DatasetTutorial| replace:: :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`
.. |Trial| replace:: :class:`~tune.concepts.flow.trial.Trial`
.. |SortMetric| replace:: :meth:`tune.concepts.flow.report.TrialReport.sort_metric`
.. |TrialObject| replace:: :class:`~tune.concepts.flow.trial.Trial`
.. |TrialReportObject| replace:: :class:`~tune.concepts.flow.report.TrialReport`
.. |LoggerLikeObject| replace:: :class:`~tune.concepts.logger.MetricLogger` object or a function producing it
.. |NonIterativeObjective| replace:: a simple python function or :class:`~tune.noniterative.objective.NonIterativeObjectiveFunc` compatible object, please read :ref:`Non-Iterative Objective Explained </notebooks/noniterative_objective.ipynb>`
.. |NonIterativeOptimizer| replace:: an object that can be converted to :class:`~tune.noniterative.objective.NonIterativeObjectiveLocalOptimizer`, please read :ref:`Non-Iterative Optimizers </notebooks/noniterative_optimizers.ipynb#Factory-Method>`
.. |DataFrameLike| replace:: Pandas, Spark, Dask or any dataframe that can be converted to Fugue :class:`~fugue.dataframe.dataframe.DataFrame`
.. |TempPath| replace:: temp path for serialized dataframe partitions. It can be empty if you preset using ``TUNE_OBJECT_FACTORY.``:meth:`~tune.api.factory.TuneObjectFactory.set_temp_path`. For details, read :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`

.. |SchemaLikeObject| replace:: :ref:`Schema like object <tutorial:/tutorials/advanced/x-like.ipynb#schema>`
.. |ParamsLikeObject| replace:: :ref:`Parameters like object <tutorial:/tutorials/advanced/x-like.ipynb#parameters>`
.. |DataFrameLikeObject| replace:: :ref:`DataFrame like object <tutorial:/tutorials/advanced/x-like.ipynb#dataframe>`
.. |DataFramesLikeObject| replace:: :ref:`DataFrames like object <tutorial:/tutorials/advanced/x-like.ipynb#dataframes>`
.. |PartitionLikeObject| replace:: :ref:`Partition like object <tutorial:/tutorials/advanced/x-like.ipynb#partition>`
.. |RPCHandlerLikeObject| replace:: :ref:`RPChandler like object <tutorial:/tutorials/advanced/x-like.ipynb#rpc>`

.. |ExecutionEngine| replace:: :class:`~fugue.execution.execution_engine.ExecutionEngine`
.. |NativeExecutionEngine| replace:: :class:`~fugue.execution.native_execution_engine.NativeExecutionEngine`
.. |FugueWorkflow| replace:: :class:`~fugue.workflow.workflow.FugueWorkflow`

.. |ReadJoin| replace:: Read Join tutorials on :ref:`workflow <tutorial:/tutorials/advanced/dag.ipynb#join>` and :ref:`engine <tutorial:/tutorials/advanced/execution_engine.ipynb#join>` for details
.. |FugueConfig| replace:: :ref:`the Fugue Configuration Tutorial <tutorial:/tutorials/advanced/useful_config.ipynb>`
.. |PartitionTutorial| replace:: :ref:`the Partition Tutorial <tutorial:/tutorials/advanced/partition.ipynb>`
.. |FugueSQLTutorial| replace:: :ref:`the Fugue SQL Tutorial <tutorial:/tutorials/fugue_sql/index.md>`
.. |DataFrameTutorial| replace:: :ref:`the DataFrame Tutorial <tutorial:/tutorials/advanced/schema_dataframes.ipynb#dataframe>`
.. |ExecutionEngineTutorial| replace:: :ref:`the ExecutionEngine Tutorial <tutorial:/tutorials/advanced/execution_engine.ipynb>`

The Space Concept
-----------------

Space
~~~~~

.. autoclass:: tune.concepts.space.spaces.Space
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

TuningParametersTemplate
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: tune.concepts.space.parameters.TuningParametersTemplate
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

TransitionChoice
~~~~~~~~~~~~~~~~

.. autoclass:: tune.concepts.space.parameters.TransitionChoice
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

Level 2 Optimizers
------------------

Hyperopt
~~~~~~~~

.. autoclass:: tune_hyperopt.optimizer.HyperoptLocalOptimizer
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Optuna
~~~~~~~~

.. autoclass:: tune_optuna.optimizer.OptunaLocalOptimizer
   :members:
   :undoc-members:
   :show-inheritance:
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
