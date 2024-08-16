# Release Notes

## 0.1.6

-  Support Python 3.12

## 0.1.5

-  Refactor `FunctionWrapper`, remove the Fugue contraint

## 0.1.3

-  Added Fugue version constraint to avoid breaking changes

## 0.1.2

-  [54](https://github.com/fugue-project/tune/pull/54): Integrated with MLFlow
-  [60](https://github.com/fugue-project/tune/issues/60): Integrated with W&B
-  [57](https://github.com/fugue-project/tune/issues/57): Expanded test coverage from Python 3.6 - 3.10, Linux and Windows
-  [58](https://github.com/fugue-project/tune/issues/58): Fixed Bayesian optimizations min_better bug

## 0.1.0

-  Replace all `pickle` with `cloudpickle`

## 0.0.9

-   Major refactoring on Space to use [parameters template](https://github.com/fugue-project/tune/issues/47)
-   Add [TransitionChoice](https://github.com/fugue-project/tune/issues/46)
-   Enable local optimizers (optuna and hyperopt) to handle [nested data structures](https://github.com/fugue-project/tune/issues/44)

## 0.0.8

-   Fixed the [lower bound](https://github.com/fugue-project/tune/issues/43) of Rand expression

## 0.0.7

-   [Optuna integration](https://github.com/fugue-project/tune/issues/23) and make Optuna and Hyperopt consistent
-   Make test coverage [100](https://github.com/fugue-project/tune/issues/16)

## 0.0.6

-   [Early stop](https://github.com/fugue-project/tune/issues/22) for non iterative
-   Work on local optimizer [1](https://github.com/fugue-project/tune/issues/18) [2](https://github.com/fugue-project/tune/issues/31)
-   Added initialize and finalize [hooks](https://github.com/fugue-project/tune/issues/28) for monitors
-   Improved realtime chart [rendering](https://github.com/fugue-project/tune/issues/19)
-   Fixed [prerelease](https://github.com/fugue-project/tune/issues/27)
-   Fixed report [timeout bug](https://github.com/fugue-project/tune/issues/20)

## Before 0.0.6

-   Implemented main features for iterative and non iterative tuning
