"""mlx.core is imported at most once per process -- PLANNING.md §5 C-mlx-cuda-no-device-abort.

With mlx's CUDA backend installed and no usable device (driver module missing after a
kernel update, CUDA_VISIBLE_DEVICES="", a container without --gpus), `import mlx.core`
raises ImportError -- but only after registering its nanobind types, so a SECOND attempt
aborts the interpreter (SIGABRT, exit 134).  polars2svg used to make two attempts
(od_flow_layout, then tfdp_layout), so `import polars2svg` itself killed the process.

Both tests run in a subprocess, because the failure mode is the process dying.  The
probe also purges and re-imports the package, so "once" is checked per process.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import polars2svg

_PKG_PARENT_ = str(Path(polars2svg.__file__).resolve().parent.parent)

# What the subprocess reports: whether the package imported, what od_flow_layout saw,
# whether TFDPLayout was (correctly) left out, and how tfdp_layout fails when asked for.
_PROBE_ = textwrap.dedent('''
    import polars2svg, polars2svg.od_flow_layout as ofl
    print('mx_is_none', ofl.mx is None)
    print('has_tfdp', hasattr(polars2svg, 'TFDPLayout'))
    try:
        import polars2svg.tfdp_layout
        print('tfdp_import', 'ok')
    except ImportError as e:
        print('tfdp_import', 'ImportError')
        print('tfdp_cause', type(e.__cause__).__name__, str(e.__cause__)[:120])
    # Purge and re-import, as importlib.reload or test_optional_dependency_extras does:
    # "once" must hold per process, not per execution of od_flow_layout.
    import sys
    for _n_ in [n for n in sys.modules if n == 'polars2svg' or n.startswith('polars2svg.')]:
        del sys.modules[_n_]
    import polars2svg
    print('reimport', 'ok', polars2svg.od_flow_layout.mx is None)
''')


def _run(env_extra: dict) -> subprocess.CompletedProcess:
    _env_ = dict(os.environ)
    _env_.update(env_extra)
    return subprocess.run([sys.executable, '-c', _PROBE_], env=_env_, capture_output=True,
                          text=True, timeout=120, cwd=_PKG_PARENT_)


def _cuda_backend_installed() -> bool:
    for _dist_ in ('mlx-cuda-13', 'mlx-cuda-12'):
        try:
            version(_dist_)
            return True
        except PackageNotFoundError:
            pass
    return False


class TestFailedMlxImportIsNotRetried(unittest.TestCase):
    """A stand-in `mlx` whose core fails like the CUDA build does on its first import and
    hard-exits with 134 -- the real abort's exit status -- on any second one.  Runs on
    every platform, mlx installed or not, since the stand-in shadows any real mlx."""

    def test_package_imports_and_tfdp_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as _tmp_:
            _mlx_ = Path(_tmp_) / 'mlx'
            _mlx_.mkdir()
            (_mlx_ / '__init__.py').write_text('')
            _count_ = Path(_tmp_) / 'attempts'
            (_mlx_ / 'core.py').write_text(textwrap.dedent(f'''
                import os
                _p_ = {str(_count_)!r}
                _n_ = (int(open(_p_).read()) if os.path.exists(_p_) else 0) + 1
                open(_p_, 'w').write(str(_n_))
                if _n_ > 1:
                    os._exit(134)
                raise ImportError('stand-in: no CUDA-capable device is detected')
            '''))
            _pp_ = os.pathsep.join([_tmp_, _PKG_PARENT_, os.environ.get('PYTHONPATH', '')])
            _r_ = _run({'PYTHONPATH': _pp_})
            _attempts_ = int(_count_.read_text()) if _count_.exists() else 0

        self.assertEqual(_r_.returncode, 0, f'the process died (exit {_r_.returncode}):\n{_r_.stderr[-1500:]}')
        self.assertEqual(_attempts_, 1, 'mlx.core must be imported exactly once')
        self.assertIn('mx_is_none True', _r_.stdout)
        self.assertIn('has_tfdp False', _r_.stdout)
        self.assertIn('tfdp_import ImportError', _r_.stdout)
        self.assertIn('reimport ok True', _r_.stdout)
        # The original failure stays visible as the cause, not swallowed.
        self.assertIn('stand-in: no CUDA-capable device', _r_.stdout)


@unittest.skipUnless(_cuda_backend_installed(), 'needs mlx-cuda-12 or mlx-cuda-13 installed')
class TestHiddenGpuDoesNotAbort(unittest.TestCase):
    """The real case, on a host with mlx's CUDA backend: hide every device and import."""

    def test_import_polars2svg_with_no_visible_cuda_device(self):
        _r_ = _run({'CUDA_VISIBLE_DEVICES': ''})
        self.assertEqual(_r_.returncode, 0, f'the process died (exit {_r_.returncode}):\n{_r_.stderr[-1500:]}')
        self.assertIn('mx_is_none', _r_.stdout)
        self.assertIn('reimport ok', _r_.stdout)
        # If a future mlx imports without a device, TFDPLayout may legitimately be present;
        # what must hold either way is that the import survived and reported consistently.
        if 'mx_is_none True' in _r_.stdout:
            self.assertIn('has_tfdp False', _r_.stdout)
            self.assertIn('tfdp_import ImportError', _r_.stdout)


if __name__ == '__main__':
    unittest.main()
