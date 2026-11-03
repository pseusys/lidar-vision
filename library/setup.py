from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension, build_ext

VERSION = "0.0.1"
DROW_DATA       = "https://github.com/VisualComputingInstitute/DROW/releases/download/v2/DROWv2-data.zip"
DROW_WEIGHTS    = "https://github.com/VisualComputingInstitute/DROW/releases/download/v2/final-WNet3xLF2p-T5-odom.rot-trainval-50ep.pth.tar"
DRSPAAM_WEIGHTS = "https://github.com/VisualComputingInstitute/DR-SPAAM-Detector/releases/download/v1.1/dr_spaam_e40.pth"
LFE_PEAKS_WEIGHTS = "https://robotics.upo.es/~famozur/onnx/LFE-Peaks.onnx"
LFE_PPN_WEIGHTS   = "https://robotics.upo.es/~famozur/onnx/LFE-PPN.onnx"
# Li2Former's own implementation, pinned. Downloaded and never vendored: it is GPL-3.0 and this
# package is MIT, so copying it into the tree would relicense the package. Only the model and the
# scan utilities are kept -- the rest of that repository is a ROS simulation workspace.
LI2FORMER_COMMIT = "a19e134b7b14f8757354ce537c006173c07575f4"
LI2FORMER_SOURCE = f"https://github.com/ai-winter/Li2Former-v1/archive/{LI2FORMER_COMMIT}.zip"

ext_modules = [
    Pybind11Extension(
        "takheliped.cpp_binding",
        sources=[str(file) for file in Path("cpp_core/sources").glob("*.cpp")],
        include_dirs=[str(Path("cpp_core/include/takheliped"))],
        define_macros=[("VERSION_INFO", VERSION)],
    ),
]



include_dir = Path("takheliped/include")
if not include_dir.exists():
    include_dir.mkdir(parents=True)
    file, _ = urlretrieve(DROW_DATA)
    with ZipFile(file, "r") as archive:
        archive.extractall(include_dir)
    urlretrieve(DROW_WEIGHTS, include_dir / Path("weights.pth.tar"))
    urlretrieve(DRSPAAM_WEIGHTS, include_dir / Path("drspaam_weights.pth"))
    urlretrieve(LFE_PEAKS_WEIGHTS, include_dir / Path("lfe_peaks.onnx"))
    urlretrieve(LFE_PPN_WEIGHTS,   include_dir / Path("lfe_ppn.onnx"))
    open(include_dir / Path("__init__.py"), "w").close()

# Kept out of the block above so an install that already has include/ still gets it.
li2former_dir = include_dir / "li2former_upstream"
if not li2former_dir.exists():
    prefix = f"Li2Former-v1-{LI2FORMER_COMMIT}/src/pedestrian_tracker/scripts/li2former/"
    file, _ = urlretrieve(LI2FORMER_SOURCE)
    with ZipFile(file, "r") as archive:
        for name in archive.namelist():
            if name.startswith(prefix) and name.endswith(".py"):
                target = li2former_dir / name[len(prefix):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(name))
        (li2former_dir / "LICENSE").write_bytes(
            archive.read(f"Li2Former-v1-{LI2FORMER_COMMIT}/LICENSE"))
    # Their transformer/ ships without one, and the adapter imports it as a package.
    for package in (li2former_dir, *(p.parent for p in li2former_dir.rglob("*.py"))):
        open(package / Path("__init__.py"), "a").close()

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
