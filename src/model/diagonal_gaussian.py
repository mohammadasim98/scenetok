from math import exp, log, pi
from typing import Optional

from jaxtyping import Float
import torch
from torch import Tensor
import numpy as np

class DiagonalGaussian:
    std_inverval: tuple[float, float]
    var_interval: tuple[float, float]
    logvar_interval: tuple[float, float]
    mean: Float[Tensor, "*batch"]
    _logvar: Float[Tensor, "*#batch"] | None = None
    _std: Float[Tensor, "*#batch"] | None = None
    _var: Float[Tensor, "*#batch"] | None = None

    def __init__(
        self,
        mean: Float[Tensor, "*batch"],
        std: Float[Tensor, "*#batch"] | None = None,
        var: Float[Tensor, "*#batch"] | None = None,
        logvar: Float[Tensor, "*#batch"] | None = None,
        logvar_interval: tuple[float, float] = (-30.0, 20.0)
    ):  
        assert sum(map(lambda x: int(x is not None), (std, var, logvar))) <= 1
        self.std_inverval = tuple(exp(0.5 * i) for i in logvar_interval)
        self.var_interval = tuple(exp(i) for i in logvar_interval)
        self.logvar_interval = logvar_interval
        self.mean = mean
        if std is not None:
            self.std = std
        if var is not None:
            self.var = var
        if logvar is not None:
            self.logvar = logvar

    @property
    def std(self) -> Float[Tensor, "*batch"]:
        if self._std is None:
            if self._var is not None:
                self._std = torch.sqrt(self._var)
            elif self._logvar is not None:
                self._std = torch.exp(0.5 * self._logvar)
            else:
                return torch.zeros(
                    (1,), device=self.device, dtype=self.dtype
                ).expand_as(self.mean)
        return self._std
    
    @std.setter
    def std(self, val: Float[Tensor, "*batch"] | None) -> None:
        self._std = val if val is None else torch.clamp(val, *self.std_inverval)
        self._var = self._logvar = None

    @property
    def var(self) -> Float[Tensor, "*batch"]:
        if self._var is None:
            if self._std is not None:
                self._var = self._std ** 2
            elif self._logvar is not None:
                self._var = torch.exp(self._logvar)
            else:
                return torch.zeros(
                    (1,), device=self.device, dtype=self.dtype
                ).expand_as(self.mean)
        return self._var

    @var.setter
    def var(self, val: Float[Tensor, "*batch"]) -> None:
        self._var = val if val is None else torch.clamp(val, *self.var_interval)
        self._std = self._logvar = None

    @property
    def logvar(self) -> Float[Tensor, "*batch"]:
        if self._logvar is None:
            if self._var is not None:
                self._logvar = torch.log(self._var)
            elif self._std is not None:
                self._logvar = 2 * torch.log(self._std)
            else:
                raise RuntimeError(
                    "Tried accessing logvar of Gaussian with zero variance"
                )
        return self._logvar
    
    @logvar.setter
    def logvar(self, val: Float[Tensor, "*batch"]) -> None:
        self._logvar = val if val is None else torch.clamp(val, *self.logvar_interval)
        self._std = self._var = None

    @property
    def device(self) -> torch.device:
        return self.mean.device
    
    @property
    def dtype(self) -> torch.dtype:
        return self.mean.dtype

    def mean_detach_(self) -> None:
        self.mean = self.mean.detach()

    def std_detach_(self) -> None:
        if self._std is not None:
            self._std = self._std.detach()
        if self._var is not None:
            self._var = self._var.detach()
        if self._logvar is not None:
            self._logvar = self._logvar.detach()

    def sample(
        self, 
        eps: Float[Tensor, "*#batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if eps is None:
            eps = torch.randn_like(self.mean)
        return self.mean + self.std * eps

    def mode(self) -> Float[Tensor, "*batch"]:
        return self.mean

    def kl(
        self, 
        other: Optional['DiagonalGaussian'] = None
    ) -> Float[Tensor, "*batch"]:
        if other is None:
            return 0.5 * (self.mean ** 2 + self.var - self.logvar - 1.0)
        logvar_delta = self.logvar - other.logvar
        return 0.5 * (
            (self.mean - other.mean) ** 2 / other.var 
            + torch.exp(logvar_delta) 
            - logvar_delta 
            - 1.0
        )

    def nll(self, sample: Tensor) -> Tensor:
        return 0.5 * (
            log(2.0 * pi) + self.logvar \
                + (sample - self.mean) ** 2 / self.var
        )

    @staticmethod
    def approx_standard_normal_cdf(x):
        """
        A fast approximation of the cumulative distribution function of the standard normal.
        """
        return 0.5 * (1.0 + torch.tanh((2.0 / torch.pi) ** 0.5 * (x + 0.044715 * torch.pow(x, 3))))
    
    def discretized_log_likelihood(
        self, 
        sample: Float[Tensor, "*batch"],
    ) -> Float[Tensor, "*batch"]:
        """
        Compute the log-likelihood of a Gaussian distribution discretizing to a given image.
        It is assumed that this was uint8 values, rescaled to the range [-1, 1].
        Returns a tensor like mean of log probabilities (in nats).
        """
        centered_x = sample - self.mean
        plus_in = (centered_x + 1.0 / 255.0) / self.std
        cdf_plus = self.approx_standard_normal_cdf(plus_in)
        min_in = (centered_x - 1.0 / 255.0) / self.std
        cdf_min = self.approx_standard_normal_cdf(min_in)
        log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min
        log_probs = torch.where(
            sample < -0.999,
            log_cdf_plus,
            torch.where(sample > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
        )
        return log_probs
    


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean