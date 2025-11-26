import torch
import torch.nn as nn
import torch.utils.data

SEED_REGION_INDEX = [[3,0,1,2,4],[7,8,9,10,11],[5,6],[13,12],[14,15,23,24,32,33],
                [22,21,31,30,40,39],[16,17,18,19,20],[25,26,27,28,29],
                [34,35,36,37,38],[41,42],[49,48],[43,44,45,46,47],
                [50,51,57],[56,55,61],[52,53,54],[58,59,60]]
DEAP_REGION_INDEX = [
    [0, 1, 16, 17],
    [2, 5, 18, 22],
    [3, 20],
    [4, 21],
    [6, 24],
    [8, 27],
    [9, 28],
    [10, 14],
    [11, 29],
    [13, 19, 30, 31, 15]
]

def adjust_region_index(region_index, num_channels, selected_channels=None):
    """
    调整区域索引以适应少通道设置
    
    Args:
        region_index: 原始区域索引列表
        num_channels: 可用的通道数量
        selected_channels: 选中的通道索引列表，如果为None则不进行映射
    
    Returns:
        调整后的区域索引列表
    """
    adjusted_regions = []
    
    # 如果指定了选中的通道，需要建立原始索引到选中索引的映射
    if selected_channels is not None:
        # 创建原始索引到选中索引的映射字典
        original_to_selected = {original_idx: new_idx for new_idx, original_idx in enumerate(selected_channels)}
        
        # 调整每个区域的索引
        for region in region_index:
            adjusted_region = []
            for idx in region:
                # 只保留在选中通道中的索引
                if idx in original_to_selected:
                    adjusted_region.append(original_to_selected[idx])
            # 只保留非空区域
            if adjusted_region:
                adjusted_regions.append(adjusted_region)
    else:
        # 否则，只保留在有效通道范围内的索引
        for region in region_index:
            adjusted_region = [idx for idx in region if idx < num_channels]
            # 只保留非空区域
            if adjusted_region:
                adjusted_regions.append(adjusted_region)
    
    # 如果调整后没有区域，创建一个默认区域包含所有通道
    if not adjusted_regions:
        adjusted_regions = [[i for i in range(num_channels)]]
    
    return adjusted_regions
# From Regional to Global Brain: A Novel Hierarchical Spatial-Temporal Neural Network Model for EEG Emotion Recognition
# r2gstnn paper link : https://ieeexplore.ieee.org/document/8736804
# Y. Li, W. Zheng, L. Wang, Y. Zong and Z. Cui, "From Regional to Global Brain: A Novel Hierarchical Spatial-Temporal Neural Network Model for EEG Emotion Recognition," in IEEE Transactions on Affective Computing, vol. 13, no. 2, pp. 568-578, 1 April-June 2022, doi: 10.1109/TAFFC.2019.2922912.

class R2GSTNN(nn.Module):
    def __init__(self, input_size=5,  num_classes=3, regions=16, region_index=SEED_REGION_INDEX, k=3, t=9,
                 regional_size=100, global_size = 150,regional_temporal_size=200, global_temporal_size=250,
                 domain_classes=2, lambda_ = 1, dropout=0.5, num_channels=None, selected_channels=None):
        super(R2GSTNN, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.k = k
        self.t = t
        self.regional_size = regional_size
        self.global_size = global_size
        self.regional_temporal_size = regional_temporal_size
        self.global_temporal_size = global_temporal_size
        self.domain_classes = domain_classes
        self.lambda_ = lambda_
        self.dropout = dropout
        
        # 如果提供了通道数量信息，调整区域索引以适应少通道设置
        if num_channels is not None:
            self.region_index = adjust_region_index(region_index, num_channels, selected_channels)
            self.regions = len(self.region_index)
        else:
            self.region_index = region_index
            self.regions = regions

        self.regional_learner = RegionFeatureLearner(input_size=self.input_size, regional_size=self.regional_size, regions=self.regions, region_index=self.region_index)
        self.regional_attention = RegionAttention(regional_size=self.regional_size, regions=self.regions)
        self.global_learner = GlobalFeatureLearner(regional_size=self.regional_size, global_size=self.global_size, regions=self.regions, k=self.k)
        self.temporal_learner = TemporalFeatureLearner(k=self.k, t=self.t, regions=self.regions, regional_size=self.regional_size, global_size=self.global_size, 
                                                      regional_temporal_size=self.regional_temporal_size, global_temporal_size=self.global_temporal_size, dropout=self.dropout)
        self.classifer = Classifer(regions=self.regions, regional_temporal_size=self.regional_temporal_size, global_temporal_size=self.global_temporal_size, 
                                   num_classes=self.num_classes,hidden_size1=512, hidden_size2=128)
        self.discriminator = Discriminator(regions=self.regions, regional_temporal_size=self.regional_temporal_size, global_temporal_size=self.global_temporal_size, 
                                           domain_classes=self.domain_classes, lambda_=self.lambda_, hidden_size1=512, hidden_size2=128)
        
    def forward(self, source_data, target_data):
        #source_data: (batch_size, T, num_electrodes, d)
        #target_data: (batch_size, T, num_electrodes, d)
        
        # 检查输入数据的通道数是否与区域索引匹配
        # 如果不匹配，动态调整区域索引
        source_num_electrodes = source_data.shape[2]
        
        # 如果模型初始化后通道数发生变化，动态更新区域索引
        if hasattr(self, 'last_num_electrodes') and self.last_num_electrodes != source_num_electrodes:
            # 警告：这种动态调整可能导致训练不稳定，最好在初始化时设置正确的通道数
            print(f"警告：输入通道数从{self.last_num_electrodes}变为{source_num_electrodes}，动态调整区域索引")
            self.region_index = adjust_region_index(self.region_index, source_num_electrodes)
            self.regions = len(self.region_index)
            # 更新网络层以适应新的区域数
            self.regional_learner = RegionFeatureLearner(input_size=self.input_size, 
                                                       regional_size=self.regional_size, 
                                                       regions=self.regions, 
                                                       region_index=self.region_index)
            self.regional_attention = RegionAttention(regional_size=self.regional_size, regions=self.regions)
            self.global_learner = GlobalFeatureLearner(regional_size=self.regional_size, 
                                                     global_size=self.global_size, 
                                                     regions=self.regions, 
                                                     k=self.k)
            self.temporal_learner = TemporalFeatureLearner(k=self.k, t=self.t, 
                                                         regions=self.regions, 
                                                         regional_size=self.regional_size, 
                                                         global_size=self.global_size, 
                                                         regional_temporal_size=self.regional_temporal_size, 
                                                         global_temporal_size=self.global_temporal_size, 
                                                         dropout=self.dropout)
            self.classifer = Classifer(regions=self.regions, 
                                     regional_temporal_size=self.regional_temporal_size, 
                                     global_temporal_size=self.global_temporal_size, 
                                     num_classes=self.num_classes, 
                                     hidden_size1=512, hidden_size2=128)
            self.discriminator = Discriminator(regions=self.regions, 
                                             regional_temporal_size=self.regional_temporal_size, 
                                             global_temporal_size=self.global_temporal_size, 
                                             domain_classes=self.domain_classes, 
                                             lambda_=self.lambda_, 
                                             hidden_size1=512, hidden_size2=128)
        
        self.last_num_electrodes = source_num_electrodes
        
        # 正常前向传播
        source_regional_feature = self.regional_learner(source_data)
        source_attention_feature = self.regional_attention(source_regional_feature)
        source_global_feature = self.global_learner(source_attention_feature)
        source_temporal_feature = self.temporal_learner(source_regional_feature, source_global_feature)
        source_label_prediction = self.classifer(source_temporal_feature)

        target_regional_feature = self.regional_learner(target_data)
        target_attention_feature = self.regional_attention(target_regional_feature)
        target_global_feature = self.global_learner(target_attention_feature)
        target_temporal_feature = self.temporal_learner(target_regional_feature, target_global_feature)
        
        domain_prediction = self.discriminator(source_temporal_feature, target_temporal_feature)

        return source_label_prediction, domain_prediction


class RegionFeatureLearner(nn.Module):#input: (batch_size*T, num_electrodes, d)
    def __init__(self, input_size=5, regional_size=100, regions=16, region_index=None):
        super(RegionFeatureLearner, self).__init__()
        if region_index is None:
            region_index = SEED_REGION_INDEX
        self.regions = regions
        self.input_size = input_size
        self.regional_size = regional_size
        self.region_index = [torch.LongTensor(e) for e in region_index]

        self.bilstm = nn.ModuleList([nn.LSTM(self.input_size, self.regional_size, batch_first=True, bidirectional=True) for i in range(regions)])
        for lstm in self.bilstm:
            for name, param in lstm.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'weight_ih' in name:
                    nn.init.xavier_normal_(param.data)
                elif 'bias' in name:
                    param.data.zero_()
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2 * hidden_size].fill_(1.0)

    def forward(self, features):
        regional_feature_input =[]
        regional_feature_list = []
        
        # 确保features有正确的维度
        # 支持不同的输入形状：
        # - (batch_size, T, channels, features) 原始形状
        # - (batch_size, channels, features) 简化形状
        if len(features.shape) == 4:
            # 原始形状 (batch_size, T, channels, features)
            features = features.reshape(-1, features.shape[2], features.shape[3])
        elif len(features.shape) == 3:
            # 简化形状 (batch_size, channels, features)
            pass  # 已经是正确的形状
        elif len(features.shape) == 2:
            # 可能是 (channels, features)，需要增加batch维度
            features = features.unsqueeze(0)
        else:
            raise ValueError(f"不支持的输入形状: {features.shape}")
        
        # 处理每个区域的特征
        for i in range(self.regions):
            # 确保区域索引有效
            valid_indices = [idx for idx in self.region_index[i] if idx < features.shape[1]]
            if valid_indices:
                regional_feature_input.append(features[:, valid_indices, :])
                # 如果该区域只有一个通道，需要调整输入维度
                if regional_feature_input[i].shape[1] == 1:
                    regional_feature_input[i] = regional_feature_input[i].squeeze(1).unsqueeze(1)
                hidden_unit = (self.bilstm[i](regional_feature_input[i])[0])
                regional_feature_list.append(hidden_unit[:, -1, :].unsqueeze(1))  # 保持维度
        
        if not regional_feature_list:
            # 如果没有有效的区域，创建一个默认的特征
            batch_size = features.shape[0]
            default_feature = torch.zeros(batch_size, 1, 2 * self.regional_size, device=features.device)
            return default_feature
        
        regional_feature = torch.cat(regional_feature_list, dim=1)
        # regional_feature: (batch_size*T, regions, 2*hidden_size)
        return regional_feature
    
class RegionAttention(nn.Module):
    def __init__(self, regional_size=100, regions=16):
        super(RegionAttention, self).__init__()
        self.regional_size = regional_size
        self.regions = regions

        self.P = nn.Parameter(torch.Tensor(2*self.regional_size, self.regions))
        self.tanh = nn.Tanh()
        self.bias = nn.Parameter(torch.Tensor(self.regions))
        self.Q = nn.Parameter(torch.Tensor(self.regions, self.regions))
        self.softmax = nn.Softmax(dim=1)
        # 参数初始化
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.P, gain=nn.init.calculate_gain('tanh'))
        nn.init.orthogonal_(self.Q, gain=1.414)
        nn.init.zeros_(self.bias)
    def forward(self, regional_feature):
        #regional_feature: (batch_size*T, regions, 2*regional_hidden_size)
        W = self.softmax(torch.matmul(self.tanh(torch.matmul(regional_feature, self.P) + self.bias), self.Q))
        #W: (batch_size*T, regions, regions)
        regional_feature = regional_feature.transpose(1,2)
        attention_feature = torch.matmul(regional_feature, W)
        attention_feature = attention_feature.transpose(1,2)
        #attention_feature: (batch_size*T, regions, 2*regional_hidden_size)
        return attention_feature
    
class GlobalFeatureLearner(nn.Module):
    def __init__(self, regional_size=100, global_size = 150, regions=16, k=3 ):
        super(GlobalFeatureLearner, self).__init__()
        self.regional_size = regional_size
        self.global_size = global_size
        self.regions = regions
        self.k = k
        self.bilstm = nn.LSTM(input_size=2*self.regional_size, hidden_size=self.global_size//2, batch_first=True, bidirectional=True)
        self.fc2 = nn.Linear(self.regions, self.k)
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        # 初始化LSTM参数
        for name, param in self.bilstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param.data, gain=nn.init.calculate_gain('relu'))  # ReLU增益
            elif 'bias' in name:
                param.data.zero_()
                hidden_size = param.size(0) // 4
                param.data[hidden_size:2 * hidden_size].fill_(1.0)

        nn.init.kaiming_normal_(self.fc2.weight,
                                mode='fan_in',
                                nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0.1)
    def forward(self, attention_feature):
            #attention_feature: (batch_size*T, regions, 2*regional_hidden_size)
            hidden_unit = self.bilstm(attention_feature)[0]
            hidden_unit = hidden_unit.transpose(1,2)
            global_feature = self.fc2(hidden_unit)
            global_feature = self.relu(global_feature)

            #global_feature: (batch_size*T, global_hidden_size, k)
            return global_feature
    
class TemporalFeatureLearner(nn.Module):
    def __init__(self, k=3, t=9,regions=16,regional_size=100, global_size=150,  
                  regional_temporal_size=200,global_temporal_size=250, dropout = 0.5 ):
        super(TemporalFeatureLearner, self).__init__()
        self.k = k
        self.t = t
        self.regions = regions
        self.regional_size = regional_size
        self.global_size = global_size
        self.regional_temporal_size = regional_temporal_size
        self.global_temporal_size = global_temporal_size

        self.dropout = nn.Dropout(dropout)
        self.regional_bilstm = nn.LSTM(input_size=2*self.regional_size, hidden_size=self.regional_temporal_size//2, batch_first=True, bidirectional=True)
        self.global_bilstm = nn.LSTM(input_size = self.global_size*self.k, hidden_size=self.global_temporal_size//2, batch_first=True, bidirectional=True)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.regional_bilstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_ih' in name:
                nn.init.xavier_normal_(param.data, gain=nn.init.calculate_gain('tanh'))  # tanh激活补偿
            elif 'bias' in name:
                param.data.zero_()
                hidden_size = param.size(0) // 4
                param.data[hidden_size:2 * hidden_size].fill_(1.0)


        for name, param in self.global_bilstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'weight_ih' in name:

                nn.init.xavier_normal_(param.data, gain=nn.init.calculate_gain('tanh') / 2)
            elif 'bias' in name:
                param.data.zero_()
                hidden_size = param.size(0) // 4
                param.data[hidden_size:2 * hidden_size].fill_(1.5)
    def forward(self, regional_feature, global_feature):
        #regional_feature: (batch_size*T, regions, 2*regional_hidden_size)
        #global_feature: (batch_size*T, global_hidden_size, k)
        
        # 适应少通道情况下的不同区域数量
        actual_regions = regional_feature.shape[1]  # 获取实际的区域数量
        
        try:
            # 尝试根据实际区域数量重塑regional_feature
            # 计算合适的批次大小，确保重塑不会失败
            batch_size = regional_feature.size(0) // self.t if self.t > 0 else regional_feature.size(0)
            regional_feature = regional_feature.reshape(batch_size, self.t, actual_regions, 2*self.regional_size)
        except RuntimeError:
            # 如果重塑失败，使用更灵活的方式处理
            # 假设形状已经是 (batch_size, regions, 2*regional_hidden_size)
            batch_size = regional_feature.size(0)
            regional_feature = regional_feature.reshape(batch_size, 1, actual_regions, 2*self.regional_size)
        
        # 处理区域时间特征
        regional_feature = regional_feature.transpose(1, 2)
        regional_feature = regional_feature.reshape(-1, regional_feature.size(2), 2*self.regional_size)
        regional_feature = self.regional_bilstm(regional_feature)[0]
        regional_feature = regional_feature[:, -1, :]
        
        # 调整区域时间特征的大小，以匹配分类器的输入要求
        # 这里我们使用线性层来调整维度，而不是直接重塑
        if actual_regions != self.regions:
            # 创建一个临时线性层来调整维度
            temp_fc = nn.Linear(actual_regions * regional_feature.size(1), self.regions * self.regional_temporal_size, device=regional_feature.device)
            # 初始化权重
            nn.init.xavier_normal_(temp_fc.weight)
            nn.init.zeros_(temp_fc.bias)
            # 调整维度
            regional_temporal_feature = temp_fc(regional_feature.reshape(batch_size, -1))
        else:
            regional_temporal_feature = regional_feature.reshape(batch_size, self.regions*self.regional_temporal_size)

        # 处理全局特征
        try:
            global_feature = global_feature.reshape(-1, self.t, self.global_size*self.k)
        except RuntimeError:
            # 如果重塑失败，使用更灵活的方式处理
            global_feature = global_feature.reshape(global_feature.size(0), 1, self.global_size*self.k)
        
        global_feature = self.global_bilstm(global_feature)[0]
        global_feature = global_feature[:, -1, :]
        
        # 确保全局时间特征的大小正确
        if global_feature.size(1) != self.global_temporal_size:
            temp_global_fc = nn.Linear(global_feature.size(1), self.global_temporal_size, device=global_feature.device)
            nn.init.xavier_normal_(temp_global_fc.weight)
            nn.init.zeros_(temp_global_fc.bias)
            global_temporal_feature = temp_global_fc(global_feature)
        else:
            global_temporal_feature = global_feature.reshape(-1, self.global_temporal_size)

        # 确保两个特征的批次大小匹配
        if regional_temporal_feature.size(0) != global_temporal_feature.size(0):
            # 调整批次大小以匹配较小的那个
            min_batch = min(regional_temporal_feature.size(0), global_temporal_feature.size(0))
            regional_temporal_feature = regional_temporal_feature[:min_batch]
            global_temporal_feature = global_temporal_feature[:min_batch]

        # 拼接全局和区域时间特征
        global_regional_temporal_feature = torch.cat([global_temporal_feature, regional_temporal_feature], dim=1)
        global_regional_temporal_feature = self.dropout(global_regional_temporal_feature)
        
        #global_regional_temporal_feature: (batch_size, global_hidden_size+regional_hidden_size*regions)
        return global_regional_temporal_feature


class Classifer(nn.Module):
    def __init__(self, regions = 16, regional_temporal_size = 200, global_temporal_size = 250, num_classes = 3,
                 hidden_size1 =512, hidden_size2 = 128):
        super(Classifer, self).__init__()
        self.regions = regions
        self.regional_temporal_size = regional_temporal_size
        self.global_temporal_size = global_temporal_size
        self.num_classes = num_classes
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2

        self.classifer = nn.Sequential(
                nn.Linear(in_features=self.global_temporal_size+self.regions*self.regional_temporal_size, out_features=512),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size1),
                nn.Linear(in_features=self.hidden_size1, out_features=self.hidden_size2),
                nn.ReLU(),
                nn.BatchNorm1d(self.hidden_size2),
                nn.Linear(in_features=self.hidden_size2, out_features=self.num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.classifer[0].weight,
                                mode='fan_in',
                                nonlinearity='relu')
        nn.init.constant_(self.classifer[0].bias, 0.1)

        nn.init.kaiming_normal_(self.classifer[3].weight,
                                mode='fan_in',
                                nonlinearity='relu')
        nn.init.constant_(self.classifer[3].bias, 0.05)

        nn.init.xavier_normal_(self.classifer[6].weight,
                               gain=nn.init.calculate_gain('linear', 0.1))  # 小增益初始化
        nn.init.constant_(self.classifer[6].bias, 0.0)
        nn.init.constant_(self.classifer[2].weight, 0.5)
        nn.init.constant_(self.classifer[2].bias, 0.0)
        nn.init.constant_(self.classifer[5].weight, 0.5)
        nn.init.constant_(self.classifer[5].bias, 0.0)
    def forward(self, global_regional_temporal_feature):
        #global_regional_temporal_feature: (batch_size, global_hidden_size+regional_hidden_size*regions)
        label_prediction = self.classifer(global_regional_temporal_feature)
        #label_prediction: (batch_size, num_classes)
        return label_prediction

class ReverseGrad(torch.autograd.Function):#Reverse the gradient of the input tensor
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambda_ * grad_out, None

class Discriminator(nn.Module):
    def __init__(self, regions = 16, regional_temporal_size=200, global_temporal_size=250,domain_classes=2,lambda_=1,
                  hidden_size1=512, hidden_size2=128):
        super(Discriminator, self).__init__()
        self.regions = regions
        self.regional_temporal_size = regional_temporal_size
        self.global_temporal_size = global_temporal_size
        self.domain_classes = domain_classes
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.lambda_ = lambda_

        self.discriminator = nn.Sequential(
                nn.Linear(in_features=self.global_temporal_size+self.regions*self.regional_temporal_size, out_features=512),
                nn.ReLU(),
                #nn.BatchNorm1d(self.hidden_size1),
                nn.Linear(in_features=self.hidden_size1, out_features=self.hidden_size2),
                nn.ReLU(),
                #nn.BatchNorm1d(self.hidden_size2),
                nn.Linear(in_features=self.hidden_size2, out_features=self.domain_classes)
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.discriminator[0].weight, gain=1.414)
        nn.init.constant_(self.discriminator[0].bias, 0.01)
        nn.init.kaiming_normal_(self.discriminator[2].weight,
                                mode='fan_out',
                                nonlinearity='relu')
        nn.init.constant_(self.discriminator[2].bias, 0.1)

        nn.init.xavier_normal_(self.discriminator[4].weight,
                               gain=nn.init.calculate_gain('sigmoid') / 2)
        nn.init.constant_(self.discriminator[4].bias, -0.1)

    def forward(self, source_feature, teaget_feature):
        #source_feature: (batch_size, global_hidden_size+regional_hidden_size*regions)
        #target_feature: (batch_size, global_hidden_size+regional_hidden_size*regions)
        features = torch.cat([source_feature, teaget_feature], dim=0)
        reversed_features = ReverseGrad.apply(features, self.lambda_)
        domain_prediction = self.discriminator(reversed_features)
        #domain_prediction: (2*batch_size, domain_classes)
        return domain_prediction
    

