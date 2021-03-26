import copy
import inspect
import json
import os
import random
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, no_type_check
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
from fugue import (
    ArrayDataFrame,
    DataFrame,
    ExecutionEngine,
    FugueWorkflow,
    IterableDataFrame,
    LocalDataFrame,
    Transformer,
    WorkflowDataFrame,
    WorkflowDataFrames,
)
from fugue._utils.interfaceless import (
    FunctionWrapper,
    _ExecutionEngineParam,
    _FuncParam,
    _OtherParam,
    is_class_method,
)
from triad import ParamDict, assert_or_throw
from triad.utils.convert import get_caller_global_local_vars, to_function

from tune.constants import TUNE_TEMP_PATH
from tune.exceptions import TuneCompileError, TuneRuntimeError
from tune.space import Space
from tune.space.parameters import _decode


class Tunable(object):
    def run(self, **kwargs: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def report(self, result: Dict[str, Any]) -> None:
        self._error = float(result["error"])
        self._hp = ParamDict(result.get("hp", None))
        self._metadata = ParamDict(result.get("metadata", None))

    @property
    def error(self) -> float:
        try:
            return self._error
        except Exception:
            raise TuneRuntimeError("error is not set")

    @property
    def hp(self) -> ParamDict:
        try:
            return self._hp
        except Exception:
            raise TuneRuntimeError("hp is not set")

    @property
    def metadata(self) -> ParamDict:
        try:
            return self._metadata
        except Exception:
            raise TuneRuntimeError("metadata is not set")

    @property
    def distributable(self) -> bool:  # pragma: no cover
        return True

    @property
    def execution_engine(self) -> ExecutionEngine:
        # pylint: disable=no-member
        try:
            return self._execution_engine  # type: ignore
        except Exception:
            raise TuneRuntimeError("execution_engine is not set")

    def space(self, *args: Space, **kwargs: Any) -> "TunableWithSpace":
        space = Space(
            **{k: v for k, v in kwargs.items() if not isinstance(v, WorkflowDataFrame)}
        )
        if len(args) > 0:
            s = args[0]
            for x in args[1:]:
                s = s * x
            space = s * space
        dfs = WorkflowDataFrames(
            {k: v for k, v in kwargs.items() if isinstance(v, WorkflowDataFrame)}
        )
        return TunableWithSpace(self, space, dfs)


class SimpleTunable(Tunable):
    def tune(self, **kwargs: Any) -> Dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def run(self, **kwargs: Any) -> None:
        res = self.tune(**kwargs)
        if "hp" not in res:
            res["hp"] = kwargs
        self.report(res)


def tunable(
    func: Optional[Callable] = None,
    distributable: Optional[bool] = None,
) -> Callable[[Any], "_FuncAsTunable"]:
    def deco(func: Callable) -> "_FuncAsTunable":
        assert_or_throw(
            not is_class_method(func),
            NotImplementedError("tunable decorator can't be used on class methods"),
        )
        return _FuncAsTunable.from_func(func, distributable=distributable)

    if func is None:
        return deco
    else:
        return deco(func)


def _to_tunable(
    obj: Any,
    global_vars: Optional[Dict[str, Any]] = None,
    local_vars: Optional[Dict[str, Any]] = None,
    distributable: Optional[bool] = None,
) -> Tunable:
    global_vars, local_vars = get_caller_global_local_vars(global_vars, local_vars)

    def get_tunable() -> Tunable:
        if isinstance(obj, Tunable):
            return copy.copy(obj)
        try:
            f = to_function(obj, global_vars=global_vars, local_vars=local_vars)
            # this is for string expression of function with decorator
            if isinstance(f, Tunable):
                return copy.copy(f)
            # this is for functions without decorator
            return _FuncAsTunable.from_func(f, distributable)
        except Exception as e:
            exp = e
        raise TuneCompileError(f"{obj} is not a valid tunable function", exp)

    t = get_tunable()
    if distributable is None:
        distributable = t.distributable
    elif distributable:
        assert_or_throw(t.distributable, TuneCompileError(f"{t} is not distributable"))
    return t


class _SingleParam(_FuncParam):
    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param, "float", "s")


class _DictParam(_FuncParam):
    def __init__(self, param: Optional[inspect.Parameter]):
        super().__init__(param, "Dict[str,Any]", "d")


class _TunableWrapper(FunctionWrapper):
    def __init__(self, func: Callable):
        super().__init__(func, "^e?[^e]+$", "^[sd]$")

    def _parse_param(
        self,
        annotation: Any,
        param: Optional[inspect.Parameter],
        none_as_other: bool = True,
    ) -> _FuncParam:
        if annotation is float:
            return _SingleParam(param)
        elif annotation is Dict[str, Any]:
            return _DictParam(param)
        elif annotation is ExecutionEngine:
            return _ExecutionEngineParam(param)
        else:
            return _OtherParam(param)

    @property
    def single(self) -> bool:
        return isinstance(self._rt, _SingleParam)

    @property
    def needs_engine(self) -> bool:
        return isinstance(self._params.get_value_by_index(0), _ExecutionEngineParam)


class _FuncAsTunable(SimpleTunable):
    @no_type_check
    def tune(self, **kwargs: Any) -> Dict[str, Any]:
        # pylint: disable=no-member
        args: List[Any] = [self.execution_engine] if self._needs_engine else []
        if self._single:
            return dict(error=self._func(*args, **kwargs))
        else:
            return self._func(*args, **kwargs)

    @no_type_check
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **kwargs)

    @property
    def distributable(self) -> bool:
        return self._distributable  # type: ignore

    @no_type_check
    @staticmethod
    def from_func(
        func: Callable, distributable: Optional[bool] = None
    ) -> "_FuncAsTunable":
        t = _FuncAsTunable()
        tw = _TunableWrapper(func)
        t._func = tw._func
        t._single = tw.single
        t._needs_engine = tw.needs_engine
        if distributable is None:
            t._distributable = not tw.needs_engine
        else:
            if distributable:
                assert_or_throw(
                    not tw.needs_engine,
                    "function with ExecutionEngine can't be distributable",
                )
            t._distributable = distributable

        return t


class ObjectiveRunner(object):
    def run(
        self, tunable: Tunable, kwargs: Dict[str, Any], hp_keys: Set[str]
    ) -> Dict[str, Any]:
        tunable.run(**kwargs)
        hp = {k: v for k, v in tunable.hp.items() if k in hp_keys}
        return {"error": tunable.error, "hp": hp, "metadata": tunable.metadata}


class TunableWithSpace(object):
    def __init__(self, tunable: Tunable, space: Space, dfs: WorkflowDataFrames):
        self._tunable = copy.copy(tunable)
        self._space = space
        self._dfs = dfs

    @property
    def tunable(self) -> Tunable:
        return self._tunable

    @property
    def space(self) -> Space:
        return self._space

    @property
    def dfs(self) -> WorkflowDataFrames:
        return self._dfs

    def tune(
        self,
        workflow: Optional[FugueWorkflow] = None,
        distributable: Optional[bool] = None,
        objective_runner: Optional[ObjectiveRunner] = None,
        how: str = "inner",
        serialize_path: str = "",
        batch_size: int = 1,
        shuffle: bool = True,
        data_space_join_type: str = "cross",
    ) -> WorkflowDataFrame:
        if len(self.dfs) > 0:
            data = serialize_dfs(self.dfs, path=serialize_path, how=how)
            space_df = space_to_df(
                data.workflow,
                self.space,
                batch_size=batch_size,
                shuffle=shuffle,
            ).broadcast()
            return tune(
                data.join(space_df, how=data_space_join_type),
                tunable=self.tunable,
                distributable=distributable,
                objective_runner=objective_runner,
            )
        else:
            assert_or_throw(
                workflow is not None, TuneCompileError("workflow is required")
            )
            space_df = space_to_df(
                workflow, self.space, batch_size=batch_size, shuffle=shuffle
            ).broadcast()
            return tune(
                space_df,
                tunable=self.tunable,
                distributable=distributable,
                objective_runner=objective_runner,
            )


def tune(  # noqa: C901
    params_df: WorkflowDataFrame,
    tunable: Any,
    distributable: Optional[bool] = None,
    objective_runner: Optional[ObjectiveRunner] = None,
) -> WorkflowDataFrame:
    t = _to_tunable(  # type: ignore
        tunable, *get_caller_global_local_vars(), distributable
    )
    if distributable is None:
        distributable = t.distributable

    if objective_runner is None:
        objective_runner = ObjectiveRunner()

    # input_has: __fmin_params__:str
    # schema: *,__fmin_value__:double,__fmin_metadata__:str
    def compute_transformer(df: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for row in df:
            dfs: Dict[str, Any] = {}
            dfs_keys: Set[str] = set()
            for k, v in row.items():
                if k.startswith("__df_"):
                    key = k[len("__df_") :]
                    if v is not None:
                        dfs[key] = pd.read_parquet(v)
                    dfs_keys.add(key)
            for params in json.loads(row["__fmin_params__"]):
                p = _decode(params)
                best = objective_runner.run(  # type: ignore
                    t, dict(**dfs, **p), set(p.keys())
                )
                res = dict(row)
                res["__fmin_params__"] = json.dumps(best["hp"])
                res["__fmin_value__"] = best["error"]
                res["__fmin_metadata__"] = json.dumps(best["metadata"])
                yield res

    # input_has: __fmin_params__:str
    def compute_processor(engine: ExecutionEngine, df: DataFrame) -> DataFrame:
        def get_rows() -> Iterable[Any]:
            keys = list(df.schema.names) + ["__fmin_value__", "__fmin_metadata__"]
            for row in compute_transformer(df.as_dict_iterable()):
                yield [row[k] for k in keys]

        t._execution_engine = engine  # type:ignore
        return ArrayDataFrame(
            get_rows(), df.schema + "__fmin_value__:double,__fmin_metadata__:str"
        )

    if not distributable:
        return params_df.process(compute_processor)
    else:
        return params_df.partition(num="ROWCOUNT", algo="even").transform(
            compute_transformer
        )


def serialize_df(df: WorkflowDataFrame, name: str, path: str = "") -> WorkflowDataFrame:
    pre_partition = df.partition_spec

    def _get_temp_path(p: str, conf: ParamDict) -> str:
        if p is not None and p != "":
            return p
        return conf.get_or_throw(TUNE_TEMP_PATH, str)  # TODO: remove hard code

    if len(pre_partition.partition_by) == 0:

        def save_single_file(e: ExecutionEngine, _input: DataFrame) -> DataFrame:
            p = _get_temp_path(path, e.conf)
            fp = os.path.join(p, str(uuid4()) + ".parquet")
            e.save_df(_input, fp, force_single=True)
            return ArrayDataFrame([[fp]], f"__df_{name}:str")

        return df.process(save_single_file)
    else:

        class SavePartition(Transformer):
            def get_output_schema(self, df: DataFrame) -> Any:
                dfn = self.params.get_or_throw("name", str)
                return self.key_schema + f"__df_{dfn}:str"

            def transform(self, df: LocalDataFrame) -> LocalDataFrame:
                p = _get_temp_path(self.params.get("path", ""), self.workflow_conf)
                fp = os.path.join(p, str(uuid4()) + ".parquet")
                df.as_pandas().to_parquet(fp)
                return ArrayDataFrame(
                    [self.cursor.key_value_array + [fp]], self.output_schema
                )

        return df.transform(SavePartition, params={"path": path, "name": name})


def serialize_dfs(
    dfs: WorkflowDataFrames, how: str = "inner", path=""
) -> WorkflowDataFrame:
    assert_or_throw(dfs.has_key, "all datarames must be named")
    serialized = WorkflowDataFrames(
        {k: serialize_df(v, k, path) for k, v in dfs.items()}
    )
    wf: FugueWorkflow = dfs.get_value_by_index(0).workflow
    return wf.join(serialized, how=how)


def space_to_df(
    wf: FugueWorkflow, space: Space, batch_size: int = 1, shuffle: bool = True
) -> WorkflowDataFrame:
    def get_data() -> Iterable[List[Any]]:
        it = list(space.encode())  # type: ignore
        if shuffle:
            random.seed(0)
            random.shuffle(it)
        res: List[Any] = []
        for a in it:
            res.append(a)
            if batch_size == len(res):
                yield [json.dumps(res)]
                res = []
        if len(res) > 0:
            yield [json.dumps(res)]

    return wf.df(IterableDataFrame(get_data(), "__fmin_params__:str"))


def select_best(df: WorkflowDataFrame, top: int = 1) -> WorkflowDataFrame:
    def _top(df: pd.DataFrame, n: int) -> pd.DataFrame:
        keys = [
            k
            for k in df.columns
            if not k.startswith("__df_") and not k.startswith("__fmin_")
        ]
        if len(keys) == 0:
            return df.sort_values("__fmin_value__").head(n)
        else:
            return df.sort_values("__fmin_value__").groupby(keys).head(n)

    return df.process(_top, params=dict(n=top))


def visualize_top_n(df: WorkflowDataFrame, top: int = 0) -> None:
    if top <= 0:
        return

    def outputter(df: LocalDataFrame) -> None:
        keys = [
            k
            for k in df.schema.names
            if not k.startswith("__df_") and not k.startswith("__fmin_")
        ]

        def show(subdf: pd.DataFrame) -> None:
            if subdf.shape[0] == 0:  # pragma: no cover
                return
            subdf = subdf.sort_values("__fmin_value__").head(top)
            title = (
                json.dumps({k: str(subdf[k].iloc[0]) for k in keys})
                if len(keys) > 0
                else ""
            )
            pdf = pd.DataFrame([json.loads(x) for x in subdf["__fmin_params__"]])
            fig = plt.figure(figsize=(12, 3 * len(pdf.columns)))
            if len(keys) > 0:
                fig.suptitle(
                    title,
                    va="center",
                    size=15,
                    weight="bold",
                    y=0.93,
                )
            for i in range(len(pdf.columns)):
                ax = fig.add_subplot(len(pdf.columns), 1, i + 1)
                pdf[pdf.columns[i]].hist(ax=ax).set_title(pdf.columns[i])
                plt.subplots_adjust(hspace=0.5)

        if len(keys) == 0:
            show(df.as_pandas())
        else:
            with FugueWorkflow() as dag:
                dag.df(df).partition(by=keys).out_transform(show)

    df.output(outputter)
