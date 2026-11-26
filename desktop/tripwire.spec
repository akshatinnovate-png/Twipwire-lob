# PyInstaller spec for the TRIPWIRE-LOB backend (the `tripwire` CLI).
#   cd desktop && pyinstaller --noconfirm --distpath backend-dist --workpath build-py tripwire.spec
# Produces backend-dist/tripwire/ (one folder: fast start, no temp unpacking), which
# electron-builder copies into the app's resources/backend/.
import os

REPO = os.path.abspath(os.path.join(SPECPATH, ".."))
datas = [
    (os.path.join(REPO, "cockpit"), "cockpit"),
    (os.path.join(REPO, "data", "events.csv"), "data"),
]
try:                                    # CA bundle so live market connections verify on every OS
    import certifi
    datas.append((certifi.where(), "certifi"))
except ImportError:
    pass
for sub in ("figures",):
    p = os.path.join(REPO, "results", sub)
    if os.path.isdir(p):
        datas.append((p, os.path.join("results", sub)))
for f in ("tables.md",):
    p = os.path.join(REPO, "results", f)
    if os.path.exists(p):
        datas.append((p, "results"))

hidden = ["cockpit", "paper_desk", "track_record", "record_binance", "fetch_lobster", "fetch_crypto", "build_cache",
          "run_benchmarks", "desk_benchmarks", "crypto_benchmarks", "preflight", "report", "splice_study",
          "matplotlib.backends.backend_agg", "certifi"]

a = Analysis(
    [os.path.join(REPO, "tripwire_cli.py")],
    pathex=[REPO, os.path.join(REPO, "scripts")],
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "pytest", "IPython", "PyQt5", "PySide6"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="tripwire", console=True, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="tripwire", upx=False)
