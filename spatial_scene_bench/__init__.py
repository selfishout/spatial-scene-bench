"""Procedural 3D scenes for probing counting and relational reasoning in vision-language models."""

from .camera import Camera, orbit_cameras
from .evaluate import mean_relative_accuracy, parse_answer, score
from .questions import GeneratorConfig, Question, generate_questions
from .render import RenderOutput, render
from .scene import Scene, SceneObject, sample_scene

__version__ = "0.1.0"

__all__ = [
    "Camera", "orbit_cameras", "Scene", "SceneObject", "sample_scene", "render", "RenderOutput",
    "Question", "GeneratorConfig", "generate_questions", "parse_answer", "score", "mean_relative_accuracy",
]
