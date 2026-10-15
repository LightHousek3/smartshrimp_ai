import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.constants import CLASSES

Label = Literal["Healthy", "BG", "WSSV", "WSSV_BG"]


class Prediction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Label
    probability: float = Field(ge=0, le=1, allow_inf_nan=False)


class DiagnosisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicted_label: Label
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    predictions: list[Prediction] = Field(min_length=4, max_length=4)
    recommendation: str
    model_version: str
    processing_time_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def check_distribution(self):
        values = [item.probability for item in self.predictions]
        if {p.label for p in self.predictions} != set(CLASSES):
            raise ValueError("Exactly four distinct ontology labels required")
        if values != sorted(values, reverse=True) or not math.isclose(sum(values), 1, abs_tol=1e-6):
            raise ValueError("Probabilities must be sorted and normalized")
        if self.predicted_label != self.predictions[0].label or self.confidence != values[0]:
            raise ValueError("Top label and confidence must agree")
        return self


class ErrorResponse(BaseModel):
    error_code: str
    error_message: str
    request_id: str
