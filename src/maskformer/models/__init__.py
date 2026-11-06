"""General MaskFormer models: the Lightning wrapper, the task overrides and the sectored decoder."""

from maskformer.models.sectored import SectoredMaskFormer, SectoredMaskFormerDecoder
from maskformer.models.tasks import ConfigurableIncidenceBasedRegressionTask, RowwiseObjectHitMaskTask
from maskformer.models.wrapper import MaskformerWrapper

__all__ = [
    "ConfigurableIncidenceBasedRegressionTask",
    "MaskformerWrapper",
    "RowwiseObjectHitMaskTask",
    "SectoredMaskFormer",
    "SectoredMaskFormerDecoder",
]
