from typing import Any

from torch import Generator


from .view_sampler import ViewSampler
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_unbounded import ViewSamplerUnbounded, ViewSamplerUnboundedCfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .view_sampler_evaluation_scene import ViewSamplerEvaluationScene, ViewSamplerEvaluationSceneCfg 
from .view_sampler_evaluation_video import ViewSamplerEvaluationVideo, ViewSamplerEvaluationVideoCfg
from .view_sampler_evaluation_video_wan import ViewSamplerEvaluationVideoWan, ViewSamplerEvaluationVideoWanCfg

from ..dtypes import Stage
from ...misc.step_tracker import StepTracker

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {

    "bounded": ViewSamplerBounded,
    "unbounded": ViewSamplerUnbounded,
    "evaluation": ViewSamplerEvaluation,
    "evaluation_video": ViewSamplerEvaluationVideo,
    "evaluation_video_wan": ViewSamplerEvaluationVideoWan,
    "evaluation_scene": ViewSamplerEvaluationScene,

}

ViewSamplerCfg = (
    ViewSamplerBoundedCfg
    | ViewSamplerEvaluationCfg
    | ViewSamplerEvaluationVideoCfg
    | ViewSamplerEvaluationVideoWanCfg
    | ViewSamplerEvaluationSceneCfg
    | ViewSamplerUnboundedCfg
)


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
    generator: Generator | None = None
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        cameras_are_circular,
        step_tracker,
        generator
    )
