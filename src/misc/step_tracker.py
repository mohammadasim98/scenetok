from multiprocessing import RLock

from jaxtyping import Int64
from torch import Tensor, tensor, int64
from torch.multiprocessing import Manager


class StepTracker:
    lock: RLock
    step: Int64[Tensor, ""]

    def __init__(self, offset: int):
        self.offset = offset
        self.lock = Manager().RLock()
        self.step = tensor(offset, dtype=int64).share_memory_()

    def set_step(self, step: int) -> None:
        with self.lock:
            self.step.fill_(step + self.offset)

    def get_step(self) -> int:
        with self.lock:
            return self.step.item()
