from tune import Checkpoint
from triad import FileSystem
from pytest import raises


def test_checkpoint(tmpdir):
    fs = FileSystem().opendir(str(tmpdir))
    cp = Checkpoint(fs)
    assert 0 == len(cp)
    with raises(AssertionError):
        cp.latest
    try:
        for i in range(4):
            with cp.create() as sfs:
                sfs.writetext("a.txt", str(i))
                if i == 3:
                    raise Exception
    except Exception:
        pass
    assert 3 == len(cp)
    assert "2" == cp.latest.readtext("a.txt")
    files = fs.listdir(".")
    assert 4 == len(files)
    cp2 = Checkpoint(fs)
    assert 3 == len(cp2)
    assert "2" == cp2.latest.readtext("a.txt")
