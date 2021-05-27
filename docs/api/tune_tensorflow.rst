tune\_tensorflow 
=================

.. |SpaceTutorial| replace:: :ref:`Space Tutorial </notebooks/space.ipynb>`
.. |DatasetTutorial| replace:: :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`
.. |Trial| replace:: :class:`~tune.concepts.flow.trial.Trial`
.. |SortMetric| replace:: :meth:`tune.concepts.flow.report.TrialReport.sort_metric`
.. |TrialObject| replace:: :class:`~tune.concepts.flow.trial.Trial`
.. |TrialReportObject| replace:: :class:`~tune.concepts.flow.report.TrialReport`
.. |NonIterativeObjective| replace:: a simple python function or :class:`~tune.noniterative.objective.NonIterativeObjectiveFunc` compatible object, please read :ref:`Non-Iterative Objective Explained </notebooks/noniterative_objective.ipynb>`
.. |NonIterativeOptimizer| replace:: an object that can be converted to :class:`~tune.noniterative.objective.NonIterativeObjectiveLocalOptimizer`, please read :ref:`Non-Iterative Optimizers </notebooks/noniterative_optimizers.ipynb#Factory-Method>`
.. |DataFrameLike| replace:: Pandas, Spark, Dask or any dataframe that can be converted to Fugue :class:`~fugue.dataframe.dataframe.DataFrame`
.. |TempPath| replace:: temp path for serialized dataframe partitions. It can be empty if you preset using ``TUNE_OBJECT_FACTORY.``:meth:`~tune.api.factory.TuneObjectFactory.set_temp_path`. For details, read :ref:`TuneDataset Tutorial </notebooks/tune_dataset.ipynb>`

.. |SchemaLikeObject| replace:: :ref:`Schema like object <tutorial:/tutorials/x-like.ipynb#schema>`
.. |ParamsLikeObject| replace:: :ref:`Parameters like object <tutorial:/tutorials/x-like.ipynb#parameters>`
.. |DataFrameLikeObject| replace:: :ref:`DataFrame like object <tutorial:/tutorials/x-like.ipynb#dataframe>`
.. |DataFramesLikeObject| replace:: :ref:`DataFrames like object <tutorial:/tutorials/x-like.ipynb#dataframes>`
.. |PartitionLikeObject| replace:: :ref:`Partition like object <tutorial:/tutorials/x-like.ipynb#partition>`
.. |RPCHandlerLikeObject| replace:: :ref:`RPChandler like object <tutorial:/tutorials/x-like.ipynb#rpc>`

.. |ExecutionEngine| replace:: :class:`~fugue.execution.execution_engine.ExecutionEngine`
.. |NativeExecutionEngine| replace:: :class:`~fugue.execution.native_execution_engine.NativeExecutionEngine`
.. |FugueWorkflow| replace:: :class:`~fugue.workflow.workflow.FugueWorkflow`

.. |ReadJoin| replace:: Read Join tutorials on :ref:`workflow <tutorial:/tutorials/dag.ipynb#join>` and :ref:`engine <tutorial:/tutorials/execution_engine.ipynb#join>` for details
.. |FugueConfig| replace:: :ref:`the Fugue Configuration Tutorial <tutorial:/tutorials/useful_config.ipynb>`
.. |PartitionTutorial| replace:: :ref:`the Partition Tutorial <tutorial:/tutorials/partition.ipynb>`
.. |FugueSQLTutorial| replace:: :ref:`the Fugue SQL Tutorial <tutorial:/tutorials/sql.ipynb>`
.. |DataFrameTutorial| replace:: :ref:`the DataFrame Tutorial <tutorial:/tutorials/schema_dataframes.ipynb#dataframe>`
.. |ExecutionEngineTutorial| replace:: :ref:`the ExecutionEngine Tutorial <tutorial:/tutorials/execution_engine.ipynb>`


tune\_tensorflow.objective
--------------------------

.. automodule:: tune_tensorflow.objective
   :members:
   :undoc-members:
   :show-inheritance:

tune\_tensorflow.spec
---------------------

.. automodule:: tune_tensorflow.spec
   :members:
   :undoc-members:
   :show-inheritance:

tune\_tensorflow.suggest
------------------------

.. automodule:: tune_tensorflow.suggest
   :members:
   :undoc-members:
   :show-inheritance:

tune\_tensorflow.utils
----------------------

.. automodule:: tune_tensorflow.utils
   :members:
   :undoc-members:
   :show-inheritance:

