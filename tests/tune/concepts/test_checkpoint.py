from tune import Checkpoint
from pytest import raises
from fsspec.implementations.dirfs import DirFileSystem


def test_checkpoint(tmpdir):
    fs = DirFileSystem(str(tmpdir))
    cp = Checkpoint(fs)
    assert 0 == len(cp)
    with raises(AssertionError):
        cp.latest
    try:
        for i in range(4):
            with cp.create() as sfs:
                with sfs.open("a.txt", "w") as f:
                    f.write(str(i))
                if i == 3:
                    raise Exception
    except Exception:
        pass
    assert 3 == len(cp)
    with cp.latest.open("a.txt", "r") as f:
        assert "2" == f.read()
    files = fs.listdir(".")
    assert 4 == len(files)
    cp2 = Checkpoint(fs)
    assert 3 == len(cp2)
    with cp2.latest.open("a.txt", "r") as f:
        assert "2" == f.read()
