from dataclasses import dataclass
from typing import Union, Optional

from vimos.base import Model, Editor, Modifier, Metric
from vimos.utils import Pipeline


@dataclass
class TaskConfig:
    model: Model
    metric: Union[str, Metric]
    editor: Optional[Union[Pipeline, Editor]] = None
    modifier: Optional[Union[Pipeline, Modifier]] = None
