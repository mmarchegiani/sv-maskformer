"""General MaskFormer models: the Lightning wrapper, the task overrides and the decoder overrides."""

from maskformer.models.decoder import DynamicQueryDecoder
from maskformer.models.sectored import SectoredMaskFormer, SectoredMaskFormerDecoder
from maskformer.models.tasks import (
    ConfigurableIncidenceBasedRegressionTask,
    ConfigurableObjectRegressionTask,
    QuerySeedTask,
    RowwiseObjectHitMaskTask,
)
from maskformer.models.wrapper import MaskformerWrapper

__all__ = [
    "ConfigurableIncidenceBasedRegressionTask",
    "ConfigurableObjectRegressionTask",
    "DynamicQueryDecoder",
    "MaskformerWrapper",
    "QuerySeedTask",
    "RowwiseObjectHitMaskTask",
    "SectoredMaskFormer",
    "SectoredMaskFormerDecoder",
]
