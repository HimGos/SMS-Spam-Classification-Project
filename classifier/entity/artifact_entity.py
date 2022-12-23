from dataclasses import dataclass


@dataclass
class DataIngestionArtifact: ...


@dataclass
class DataValidationArtifact: ...


@dataclass
class DataTransformationArtifact: ...


@dataclass
class ModelTrainerArtifact: ...


@dataclass
class ModelEvaluationArtifact: ...


@dataclass
class ModelPusherArtifact: ...

