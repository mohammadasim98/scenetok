
from torch import Tensor, concat
from jaxtyping import Float, Bool
from dataclasses import dataclass
from typing import Optional, Union, Any

@dataclass
class CameraInputs:

    intrinsics: Float[Tensor, "batch view 3 3"]
    extrinsics: Float[Tensor, "batch view 4 4"]
    def __getitem__(self, index: Any) -> "CameraInputs":
        return type(self)(
            intrinsics = self.intrinsics[index],
            extrinsics = self.extrinsics[index]
        )

    def flatten(self):

        return concat([self.extrinsics.flatten(2, 3), self.intrinsics.flatten(2, 3)], dim=-1)
    def __len__(self) -> int:
        return self.intrinsics.shape[1]
@dataclass
class DenoiserInputs:

    view: Float[Tensor, "batch view _ height width"]
    pose: CameraInputs
    timestep: Float[Tensor, "batch view"]
    state: Optional[Float[Tensor, "batch num _"]]=None


@dataclass
class CompressorInputs:

    view: Float[Tensor, "batch view channel height width"]
    pose: CameraInputs
    mask: Optional[Float[Tensor, "batch view"]]=None


@dataclass
class SceneGeneratorInputs:

    view: Float[Tensor, "batch view _ height width"]
    pose: CameraInputs
    anchor_pose: CameraInputs
    timestep: Optional[Float[Tensor, "batch view"]]=None
    state: Optional[Float[Tensor, "batch num _"]]=None

