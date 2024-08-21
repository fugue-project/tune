from tune.concepts.logger import (
    MetricLogger,
    get_current_metric_logger,
    set_current_metric_logger,
)


def test_logger_context():
    m1 = MetricLogger()
    m2 = MetricLogger()

    with set_current_metric_logger(m1) as mm1:
        assert get_current_metric_logger() is m1
        assert mm1 is m1
        with set_current_metric_logger(m2) as mm2:
            assert get_current_metric_logger() is m2
            assert mm2 is m2
        assert get_current_metric_logger() is m1
