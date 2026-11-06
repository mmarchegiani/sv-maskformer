"""General MaskFormer models: the Lightning wrapper, the task overrides and the sectored decoder."""

from maskformer.models.sectored import SectoredMaskFormer, SectoredMaskFormerDecoder
from maskformer.models.tasks import ConfigurableIncidenceBasedRegressionTask, ConfigurableObjectRegressionTask, RowwiseObjectHitMaskTask
from maskformer.models.wrapper import MaskformerWrapper

__all__ = [
    "ConfigurableIncidenceBasedRegressionTask",
    "ConfigurableObjectRegressionTask",
    "MaskformerWrapper",
    "RowwiseObjectHitMaskTask",
    "SectoredMaskFormer",
    "SectoredMaskFormerDecoder",
]
