from pydantic import BaseModel, Field
from typing import Dict

class PredictionResponse(BaseModel):
    Predicted_category: str =Field(..., description='The Predicted category of Premium Insurance', exampl='Low' )
    confidence_score: float =Field(..., description="Model's confidence score for the predicted", example=0.8)
    class_probablities: Dict[str,float] = Field(..., description="Probablity of All possible classes", example={"Low":.4, "Medium":0.79, "Heigh":0.9})
