import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pyswarms as ps
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader as GDaraLoader
import spikingjelly.timing_based.encoding as encoding
from torch_geometric.data import Data
from datetime import datetime, timedelta
import json
import numpy as np
import os
import math
from torch_geometric.nn import GCNConv

import base64
from io import BytesIO
import spikingjelly.timing_based.neuron as neuron


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphSNNDataset(Dataset):
    def __init__(self, num_nodes_per_graph: int, per_dist: any, per_height: any, per_samecity: any, rainfull_except_value: any):
        """
        初始化数据集。

        :param data_list: 包含图数据的列表，每个元素是一个字典，包括节点特征、边缘索引和每个节点的标签。
        """
        self.num_nodes_per_graph = num_nodes_per_graph
        self.per_height = per_height
        self.per_dist = per_dist
        self.per_samecity = per_samecity
        self.rainfull_except_value = rainfull_except_value

        cityist = ['beijing', 'hangzhou', 'shanghai',
                   'shenzhen', 'sijiazhuang', 'wuhan']

        data_list = []
        city_index = 0

        for city in cityist:
            with open(f'../means_dataset/{city}/data.json', 'r') as file:
                city_list = json.load(file)
                city_list = self.add_cityid_to_objects(city_list, city_index)

            data_list = data_list+city_list
            city_index += 1

        self.graphdataist = self.create_graphs_from_nodes(
            data_list, self.num_nodes_per_graph)

    def __len__(self):
        """
        返回数据集中的图数量。
        """
        graph_num = len(self.graphdataist)
        return graph_num

    def __getitem__(self, idx):
        """
        根据索引获取单个图的数据。

        :param idx: 索引。
        :return: 单个图的数据。
        """
        # 处理节点特征、边缘索引和标签
        graphdata = self.graphdataist[idx]

        return graphdata

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # 将经纬度从度转换为弧度
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # 地球半径（千米）
        R = 6371.0

        # 计算纬度和经度的变化
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        # 应用haversine公式
        a = math.sin(dlat / 2)**2 + math.cos(lat1) * \
            math.cos(lat2) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # 计算总距离
        distance = R * c
        return distance

    def calculate_weight(self, dist, height_diff, samecity_true_or_false):
        # 根据距离和高度差计算权重
        return self.per_dist / (dist + 1e-6) + self.per_height / (height_diff + 1e-6) + self.per_samecity*samecity_true_or_false

    def create_graphs_from_nodes(self, node_data, num_nodes_per_graph):
        graphs = []
        # labelist = []
        for i in range(0, len(node_data) - num_nodes_per_graph + 1, num_nodes_per_graph):
            current_nodes = node_data[i:i + num_nodes_per_graph]
            x = []  # 节点特征
            edge_index = []  # 边缘索引
            edge_weight = []  # 边的权重
            labelist_per_graph = []

            for node in current_nodes:
                seq_rainfall = np.array(
                    node["24h_rainfall_sequence"])/self.rainfull_except_value
                x.append(seq_rainfall.tolist())

                labelist_per_graph.append(node['level'])

            # labelist.append(labelist_per_graph)

            # 构建边缘索引和权重
            for i in range(len(current_nodes)):
                for j in range(len(current_nodes)):
                    if i != j:
                        # counting two node dist
                        dist = self.haversine_distance(float(current_nodes[i]["lat"]), float(current_nodes[i]["lnt"]),
                                                       float(current_nodes[j]["lat"]), float(current_nodes[j]["lnt"]))
                        # counting two node height diff
                        height_diff = abs(
                            current_nodes[i]["height"] - current_nodes[j]["height"])
                        # counting two node isn't in same city
                        if current_nodes[i]['cityid'] == current_nodes[j]['cityid']:
                            samecity_true_or_false = 1
                        else:
                            samecity_true_or_false = 0
                        weight = self.calculate_weight(
                            dist, height_diff, samecity_true_or_false)
                        edge_index.append([i, j])
                        edge_weight.append(weight)

            x = torch.tensor(x, dtype=torch.float).unsqueeze(-2)

            edge_index = torch.tensor(
                edge_index, dtype=torch.long).t().contiguous()

            edge_weight = torch.tensor(edge_weight, dtype=torch.float)

            labels = torch.tensor(labelist_per_graph)

            graph = Data(x=x, edge_index=edge_index,
                         edge_attr=edge_weight, labels=labels)
            graphs.append(graph)

        # labelist = torch.tensor(labelist)
        return graphs

    def add_cityid_to_objects(self, object_array, cityid_value):
        """
        向数组中的每个对象添加一个名为 'cityid' 的属性。

        :param object_array: 包含对象的数组。
        :param cityid_value: 要设置的 'cityid' 的值。
        """
        for obj in object_array:
            # setattr(obj, 'cityid', cityid_value)
            obj['cityid'] = cityid_value
        return object_array


class GraphGCNWithSequence(nn.Module):
    def __init__(self, num_node_features: int, gcn_hidden1: int, gcn_hidden2: int, seq_length: int, num_classes: int):
        super(GraphGCNWithSequence, self).__init__()

        self.seq_length = seq_length
        self.num_classes = num_classes

        self.conv1 = GCNConv(num_node_features, gcn_hidden1)
        self.conv2 = GCNConv(gcn_hidden1, gcn_hidden2)
        self.conv3 = GCNConv(gcn_hidden2, gcn_hidden2)

        # 全连接层，用于融合所有时间步的特征
        self.fc_seq = Linear(gcn_hidden2 * seq_length, gcn_hidden2)
        # 分类层
        self.classifier = Linear(gcn_hidden2, num_classes)

    def forward(self, x, edge_index, edge_weight=None):
        # x的形状 [num_nodes, num_node_features, seq_length]
        seq_length = x.shape[-1]
        node_features_list = []

        for t in range(seq_length):
            # 对每个时间步进行图卷积
            x_t = x[:, :, t]
            x_t = self.conv1(x_t, edge_index, edge_weight=edge_weight)
            x_t = F.relu(x_t)
            x_t = self.conv2(x_t, edge_index, edge_weight=edge_weight)
            x_t = F.relu(x_t)
            x_t = self.conv3(x_t, edge_index, edge_weight=edge_weight)
            x_t = F.relu(x_t)
            node_features_list.append(x_t)

        # 融合所有时间步的特征
        node_features_seq = torch.cat(node_features_list, dim=1)
        node_features_seq = self.fc_seq(node_features_seq)
        node_features_seq = F.relu(node_features_seq)

        # 使用全连接层将节点特征转换为分类标签
        out = self.classifier(node_features_seq)

        return out


def train_gcn_snn(num_nodes_per_graph: int, per_dist: any, per_height: any, per_samecity: any, rainfull_except_value: any, batch_size: int, train_epoch: int, gcn_hidden1: int, gcn_hidden2: int,  lr: float, file_path: str):

    g_data = GraphSNNDataset(num_nodes_per_graph=num_nodes_per_graph, per_dist=per_dist,
                             per_height=per_height, per_samecity=per_samecity, rainfull_except_value=rainfull_except_value)

    # 定义训练集和测试集的比例
    train_ratio = 0.8
    total_size = g_data.__len__()
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    # 随机划分数据集
    torch.manual_seed(0)  # 设置随机种子
    train_dataset, test_dataset = random_split(g_data, [train_size, test_size])

    train_data_loader = GDaraLoader(
        dataset=train_dataset, batch_size=batch_size)

    test_data_loader = GDaraLoader(
        dataset=test_dataset, batch_size=batch_size)

    # 训练流程
    # 实例化模型

    model = GraphGCNWithSequence(num_node_features=1, gcn_hidden1=gcn_hidden1, gcn_hidden2=gcn_hidden2,
                                 seq_length=48, num_classes=3).to(device)

    # 优化器和损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 开启 PyTorch 的自动混合精度
    scaler = torch.cuda.amp.GradScaler()

    train_times = 0
    # max_test_accuracy = 0
    for epoch in range(train_epoch):
        # train_times = 0
        train_correct_sum = 0
        train_sum = 0
        # max_test_accuracy = 0
        model.train()
        for data in train_data_loader:

            optimizer.zero_grad()

            data = data.to(device)
            target = data.labels.to(device)

            # 使用自动混合精度
            with torch.cuda.amp.autocast():
                v_max = model(data.x, data.edge_index, data.edge_attr)
                loss = criterion(v_max, target)

            # 缩放损失并反向传播
            scaler.scale(loss).backward()

            # 更新模型参数
            scaler.step(optimizer)

            # 更新缩放器
            scaler.update()

            train_correct_sum += (v_max.argmax(dim=1) ==
                                  target).float().sum().item()
            train_sum += target.numel()

            train_times += 1
        train_accuracy = train_correct_sum / train_sum
        # print(train_accuracy, '\n')

    print("Testing...")
    model.eval()
    # true_levelist = []
    # forcastingist = []
    with torch.no_grad():
        correct_num = 0
        img_num = 0
        for data in test_data_loader:
            data = data.to(device)
            # x0 = data.x.unsqueeze(0).to(device)
            v_max = model(data.x, data.edge_index, data.edge_attr)

            target = data.labels.to(device)

            # print('v_max', v_max.argmax(dim=1), 'label',
            #       label.to(device))
            correct_num += (v_max.argmax(dim=1) ==
                            target).float().sum().item()

            img_num += data.x.shape[0]

        test_accuracy = correct_num / img_num
        print("Time {} Epoch {}, train_acc = {}, test_acc={},  train_times={}".format(
            datetime.now(), epoch+1, train_accuracy, test_accuracy,  train_times))
        print('------------------------------\n')

    # 假设model是你已经训练好的PyTorch模型
    state_dict = model.state_dict()

    # 创建一个将保存编码状态字典的字典
    encoded_state_dict = {}

    # 对于state_dict中的每个参数，将其转换为base64编码的字符串
    for key, value in state_dict.items():
        buffer = BytesIO()
        torch.save(value, buffer)
        encoded_state_dict[key] = base64.b64encode(
            buffer.getvalue()).decode("utf-8")
        
    # save parameters, turlevelist, forcastingist
    savefile = {
        "num_nodes_per_graph": num_nodes_per_graph,
        "per_dist": per_dist,
        "per_height": per_height,
        "per_samecity": per_samecity,
        "rainfull_except_value": rainfull_except_value,
        "batch_size": batch_size,
        "train_epoch": train_epoch,
        "gcn_hidden1": gcn_hidden1,
        "gcn_hidden2": gcn_hidden2,
        # "m": m,
        # "T": T,
        # "tau": tau,
        # "tau_s": tau_s,
        "lr": lr,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "modelbase64code": encoded_state_dict,
    }

    append_to_json_file(savefile, file_path)

    # 清理缓存
    torch.cuda.empty_cache()
    # 调用垃圾回收器
    gc.collect()
    # return loss
    return 1-test_accuracy


def append_to_json_file(data, file_path):
    # 检查文件是否存在
    if os.path.exists(file_path):
        # 读取现有数据
        with open(file_path, 'r') as file:
            try:
                file_data = json.load(file)
            except json.JSONDecodeError:
                # 如果文件为空或格式不正确，则创建一个空列表
                file_data = []
    else:
        # 如果文件不存在，创建一个空列表
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_data = []

    # 添加新数据
    file_data.append(data)

    # 写回文件
    with open(file_path, 'w') as file:
        json.dump(file_data, file, indent=4)



