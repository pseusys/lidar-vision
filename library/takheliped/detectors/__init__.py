from .detector import Detector
from .drow_detector import DrowDetector
from .algorithmic_detector import AlgorithmicDetector
from .architectures import DrSpaamDetector
from .full_scan import SpaceTimeCNNDetector
from .li2former import Li2FormerDetector
from .li2former_official import OfficialLi2FormerDetector
from .lfe_detector import LFEPeaksDetector, LFEPPNDetector

# Registry of all trainable/evaluable detectors.
# Keys are the canonical --detector CLI argument values.
# AlgorithmicDetector is included for evaluation but cannot be trained.
# DROW and DR-SPAAM use published weights and are not trained via train.py.
# Neither Li2Former entry has published weights; train from scratch or skip.
# `li2former` is this project's own reading of the paper's prose and is NOT the paper's
# architecture; `li2former_official` wraps the authors' released network, downloaded into
# include/ by setup.py. Only the latter's numbers belong in a Li2Former row.
# LFE-Peaks and LFE-PPN are ONNX-based; evaluation only.
DETECTOR_REGISTRY: dict = {
    "algorithmic":       AlgorithmicDetector,
    "drow":              DrowDetector,
    "drspaam":           DrSpaamDetector,
    "li2former":         Li2FormerDetector,
    "li2former_official": OfficialLi2FormerDetector,
    "spacetime_cnn":     SpaceTimeCNNDetector,
    # LFE detectors are ONNX-based and not registered for PyTorch training
    "lfe_peaks":         LFEPeaksDetector,
    "lfe_ppn":           LFEPPNDetector,
}
