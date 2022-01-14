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


from typing import NamedTuple

import numpy
import torch
import torchvision


class Model():
    """
    PyTorch model class
    """
    def __init__(self, model_name):
        super().__init__()
        model_func = getattr(torchvision.models, model_name)
        self._model = model_func(pretrained=True)
        state_dict = None
        if model_name == 'resnet101':
            state_dict = torch.hub.load_state_dict_from_url(
                'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1h-36d3f2aa.pth')
        if model_name == 'resnet50':
            state_dict = torch.hub.load_state_dict_from_url(
                'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth')
        if state_dict:
            self._model.load_state_dict(state_dict)

        self._model.fc = torch.nn.Identity()
        self._model.eval()

    def __call__(self, img_tensor: torch.Tensor):
        return self._model(img_tensor).flatten().detach().numpy()

    def train(self):
        """
        For training model
        """
        pass