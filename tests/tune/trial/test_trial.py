from tune.trial.trial import Trial, ReportFeedback, ReportHandler
from tune import Rand
from triad import FileSystem
from tune.checkpoint import Checkpoint


def test_report_feedback():
    f = ReportFeedback(True, False, a=1, b=2)
    assert f.should_stop
    assert not f.should_checkpoint
    assert dict(a=1, b=2) == f.metadata


def test_trial(tmpdir):
    h = ReportHandler()
    params = dict(
        a=1,
        b=Rand(0, 1),
    )
    metadata = dict(c=3)
    trial = Trial("abc", params, metadata, h, None)
    assert 0 == trial.iteration
    assert "abc" == trial.trial_id
    params["a"] = 2
    assert (
        dict(
            a=1,
            b=Rand(0, 1),
        )
        == trial.params
    )
    params["c"] = 4
    assert dict(c=3) == trial.metadata
    assert not trial.stateful

    trial2 = trial.with_params({"x": 1})
    assert trial.params != trial2.params
    assert {"x": 1} == trial2.params
    assert trial.trial_id == trial2.trial_id

    fs = FileSystem().opendir(str(tmpdir))
    trial2 = trial.with_checkpoint_basedir_fs(fs)
    assert trial2.stateful
    assert 0 == trial2.iteration
    with trial2.checkpoint.create() as fd:
        fd.writetext("a.txt", "abc")
    trial3 = trial.with_checkpoint_basedir_fs(fs)
    with trial3.checkpoint.latest as fd:
        assert "abc" == fd.readtext("a.txt")
    assert not trial.stateful

    fb = trial.report(2.0)
    assert fb.should_stop and not fb.should_checkpoint

    h2 = MockHandler()
    trial2 = trial.with_report_handler(h2)
    trial2.report(3.0)
    assert h2.trial is trial2
    assert h2.params is h2.trial.params
    assert 3.0 == h2.metric
    assert {} == h2.metadata


def test_stateful_trial(tmpdir):
    t = Trial(
        "abc",
        dict(a=1),
        dict(b=2),
        report_handler=MockHandler2(),
        checkpoint_basedir_fs=FileSystem().opendir(str(tmpdir)),
    )

    def run(trial: Trial):
        assert trial.stateful
        if len(trial.checkpoint) > 0:
            data = trial.checkpoint.latest.readtext("a.txt")
        else:
            data = ""
        while True:
            data += "i"
            fb = trial.report(
                len(data), save_checkpoint=lambda fs: fs.writetext("a.txt", data)
            )
            if fb.should_stop:
                return

    run(t)  # stop at 1
    assert "i" == t.checkpoint.latest.readtext("a.txt")
    run(t)  # stop at 5
    assert "iiiii" == t.checkpoint.latest.readtext("a.txt")


class MockHandler(ReportHandler):
    def __call__(self, trial, params, metric, metadata):
        self.trial = trial
        self.params = params
        self.metric = metric
        self.metadata = metadata
        return super().__call__(trial, params, metric, metadata)


class MockHandler2(ReportHandler):
    def __call__(self, trial, params, metric, metadata):
        if metric == 1 or metric == 5:
            return ReportFeedback(True, True)
        return ReportFeedback(False, True)
