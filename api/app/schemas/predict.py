from typing import Any, List, Optional

from pydantic import BaseModel
from cvd_model.processing.validation import DataInputSchema

class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[str]]


class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "General_Health":"Poor",
                        "Checkup":"Within the past 2 years",
                        "Exercise":"No",
                        "Skin_Cancer":"No",
                        "Other_Cancer":"No",
                        "Depression":"No",
                        "Diabetes":"No",
                        "Arthritis":"Yes",
                        "Sex":"Female",
                        "Age_Category":"70-74",
                        "Height_cm":150.0,
                        "Weight_kg":32.66,
                        "BMI":14.54,
                        "Smoking_History":"Yes",
                        "Alcohol_Consumption":0.0,
                        "Fruit_Consumption":30.0,
                        "Green_Vegetables_Consumption":16.0,
                        "FriedPotato_Consumption":12.0
                    },
                ]
            }
        }