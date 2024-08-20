from tune.concepts.logger import (
    MetricLogger,
    get_current_metric_logger,
    set_current_metric_logger,
)


def test_logger_context():
    m1 = MetricLogger()
    m2 = MetricLogger()

    with set_current_metric_logger(m1):
        assert get_current_metric_logger() == m1
        with set_current_metric_logger(m2):
            assert get_current_metric_logger() == m2
        assert get_current_metric_logger() == m1
