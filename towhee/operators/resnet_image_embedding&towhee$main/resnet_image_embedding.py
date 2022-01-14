# Copyright 2021 Zilliz. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import numpy
import torch
import torchvision
from PIL import Image
from torchvision import transforms
from pathlib import Path
from typing import NamedTuple
import os
from torchvision.transforms import InterpolationMode
from towhee.operator import Operator
from towhee.utils.pil_utils import to_pil
import warnings
warnings.filterwarnings("ignore")

class ResnetImageEmbedding(Operator):
    """
    PyTorch model for image embedding.
    """
    def __init__(self, model_name: str, framework: str = 'pytorch') -> None:
        super().__init__()
        if framework == 'pytorch':
            import importlib.util
            path = os.path.join(str(Path(__file__).parent), 'pytorch', 'model.py')
            opname = os.path.basename(str(Path(__file__))).split('.')[0]
            spec = importlib.util.spec_from_file_location(opname, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        self.model = module.Model(model_name)
        self.tfms = transforms.Compose([transforms.Resize(235, interpolation=InterpolationMode.BICUBIC),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __call__(self, image: 'towhee.types.Image') -> NamedTuple('Outputs', [('feature_vector', numpy.ndarray)]):
        img = self.tfms(to_pil(image)).unsqueeze(0)
        embedding = self.model(img)
        Outputs = NamedTuple('Outputs', [('feature_vector', numpy.ndarray)])
        return Outputs(embedding)
