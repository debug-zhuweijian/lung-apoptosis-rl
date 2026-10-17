"""Thesis V1 biological modules; provenance and data-role guards live in engine."""

from .config import ModelConfig
from .compact import CompactResponseModel
from .factory import build_response_model
from .encoders import (CellEncoder, ConditionEncoder, DoseEncoder, DrugEncoder,
                       DurationEncoder, GenerativeTimeEncoder, HistoryEncoder)
from .flow import FlowModel, PADVelocity, ResponseModel, heun_integrate
from .heads import (BackwardModel, CorrectionModel, EndpointHead,
                    MonotoneEndpointCalibrator, PopulationPool,
                    StateClassifier, TemperatureCalibrator)
from .biology import average_gene_ranks, biological_features
from .objectives import (ProjectionResult, entropic_ot, group_expression_error,
                         mean_squared_distance, physical_projection, rbf_kernel,
                         rbf_mmd, sample_ot_pairs)

__all__ = [
    "CompactResponseModel", "build_response_model", "ModelConfig", "CellEncoder", "ConditionEncoder", "DoseEncoder", "DrugEncoder",
    "DurationEncoder", "GenerativeTimeEncoder", "HistoryEncoder", "FlowModel",
    "PADVelocity", "ResponseModel", "heun_integrate", "BackwardModel", "CorrectionModel",
    "EndpointHead", "MonotoneEndpointCalibrator", "PopulationPool",
    "StateClassifier", "TemperatureCalibrator", "average_gene_ranks",
    "biological_features", "ProjectionResult", "entropic_ot", "group_expression_error",
    "mean_squared_distance", "physical_projection", "rbf_kernel", "rbf_mmd",
    "sample_ot_pairs",
]

