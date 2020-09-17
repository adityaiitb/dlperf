# Copyright (c) 2018, deepakn94, robieta. All rights reserved.
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
#
# -----------------------------------------------------------------------
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

# Copyright (c) 2020, Aditya Agrawal.
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

import numpy as np
import torch
import torch.nn as nn

import sys
from os.path import abspath, join, dirname
import torch.cuda.nvtx as nvtx

class NeuMF(nn.Module):
    def __init__(self, nb_users, nb_items,
                 mf_dim, mlp_layer_sizes, dropout=0):
        
        if mlp_layer_sizes[0] % 2 != 0:
            raise RuntimeError('u dummy, mlp_layer_sizes[0] % 2 != 0')
        super(NeuMF, self).__init__()
        nb_mlp_layers = len(mlp_layer_sizes)

        self.mf_user_embed = nn.Embedding(nb_users, mf_dim)
        self.mf_item_embed = nn.Embedding(nb_items, mf_dim)
        self.mlp_user_embed = nn.Embedding(nb_users, mlp_layer_sizes[0] // 2)
        self.mlp_item_embed = nn.Embedding(nb_items, mlp_layer_sizes[0] // 2)
        self.dropout = dropout

        self.mlp = nn.ModuleList()
        for i in range(1, nb_mlp_layers):
            self.mlp.extend([nn.Linear(mlp_layer_sizes[i - 1], mlp_layer_sizes[i])])  # noqa: E501

        self.final = nn.Linear(mlp_layer_sizes[-1] + mf_dim, 1)

        self.mf_user_embed.weight.data.normal_(0., 0.01)
        self.mf_item_embed.weight.data.normal_(0., 0.01)
        self.mlp_user_embed.weight.data.normal_(0., 0.01)
        self.mlp_item_embed.weight.data.normal_(0., 0.01)

        def glorot_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features
            limit = np.sqrt(6. / (fan_in + fan_out))
            layer.weight.data.uniform_(-limit, limit)

        def lecunn_uniform(layer):
            fan_in, fan_out = layer.in_features, layer.out_features  # noqa: F841, E501
            limit = np.sqrt(3. / fan_in)
            layer.weight.data.uniform_(-limit, limit)
        for layer in self.mlp:
            if type(layer) != nn.Linear:
                continue
            glorot_uniform(layer)
        lecunn_uniform(self.final)

    def forward(self, user, item, sigmoid=False):
        nvtx.range_push("layer:MF_User_Embedding")
        xmfu = self.mf_user_embed(user)
        nvtx.range_pop()

        nvtx.range_push("layer:MF_Item_Embedding")
        xmfi = self.mf_item_embed(item)
        nvtx.range_pop()

        nvtx.range_push("layer:GMF")
        xmf = xmfu * xmfi
        nvtx.range_pop()

        nvtx.range_push("layer:MLP_User_Embedding")
        xmlpu = self.mlp_user_embed(user)
        nvtx.range_pop()

        nvtx.range_push("layer:MLP_Item_Embedding")
        xmlpi = self.mlp_item_embed(item)
        nvtx.range_pop()

        nvtx.range_push("layer:MLP_Concat")
        xmlp = torch.cat((xmlpu, xmlpi), dim=1)
        nvtx.range_pop()

        for i, layer in enumerate(self.mlp):
            nvtx.range_push("layer:MLP_{}".format(i))
            xmlp = layer(xmlp)
            xmlp = nn.functional.relu(xmlp)

            if self.dropout != 0:
                nvtx.range_push("layer:Dropout")
                xmlp = nn.functional.dropout(xmlp, p=self.dropout, training=self.training)
                nvtx.range_pop()
            nvtx.range_pop()

        nvtx.range_push("layer:NeuMF_Concat")
        x = torch.cat((xmf, xmlp), dim=1)
        nvtx.range_pop()

        nvtx.range_push("layer:NeuMF_MLP")
        x = self.final(x)
        nvtx.range_pop()

        if sigmoid:
            nvtx.range_push("layer:Sigmoid")
            x = torch.sigmoid(x)
            nvtx.range_pop()
        return x
