from typing import Callable, Literal, TypedDict, Optional

from jaxtyping import Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]


# The following types mainly exist to make type-hinted keys show up in VS Code. Some
# dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call (e.g., the number of views,
#    which differs between context and target BatchedViews).


class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view
    latent: Optional[Float[Tensor, "batch _ _ _ _"]]=None  # batch view channel height width

    def to(self, device: str="cuda"):
        
        return type(self)(
            self.extrinsics.to(device),
            self.intrinsics.to(device),
            self.image.to(device),
            self.near.to(device),
            self.far.to(device),
            self.index.to(device),
            self.latent.to(device) if self.latent is not None else None
        )
        
class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]
    
    def to(self, device: str="cuda"):
        
        return type(self)(
            BatchedViews(self.extrinsics.to(device),
                self.target.intrinsics.to(device),
                self.target.image.to(device),
                self.target.near.to(device),
                self.target.far.to(device),
                self.target.index.to(device),
                self.target.latent.to(device) if self.target.latent is not None else None
            ),
            BatchedViews(self.extrinsics.to(device),
                self.context.intrinsics.to(device),
                self.context.image.to(device),
                self.context.near.to(device),
                self.context.far.to(device),
                self.context.index.to(device),
                self.context.latent.to(device) if self.context.latent is not None else None
            ),
            self.scene
        )


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    latent: Float[Tensor, "_ _ _ _"]  # batch view channel height width
    image: Float[Tensor, "_ 3 height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str


# A data shim modifies the example after it's been returned from the data loader.
DataShim = Callable[[BatchedExample], BatchedExample]

AnyExample = BatchedExample | UnbatchedExample
AnyViews = BatchedViews | UnbatchedViews
