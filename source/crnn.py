import torch
import torch.nn as nn
import dpn


class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden*2, nOut)
    
    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T*b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output

class base_CNN(nn.Module):
    def __init__(self, in_channels=3):
        super(base_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 2, 1, 0)
        self.maxpool_2x2 = nn.MaxPool2d((2, 2), 2)
        self.maxpool_2x1 = nn.MaxPool2d((2, 1), 2)
        self.BatchNorm2d_5_6 = nn.BatchNorm2d(512)
        self.BatchNorm2d_6_7 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
    
    def forward(self, input_tensor):
        x = self.maxpool_2x2(self.relu(self.conv1(input_tensor)))
        x = self.maxpool_2x2(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.maxpool_2x1(self.relu(self.conv4(x)))
        x = self.BatchNorm2d_5_6(self.relu(self.conv5(x)))
        x = self.BatchNorm2d_6_7(self.relu(self.conv6(x)))
        x = self.maxpool_2x1(x)
        x = self.relu(self.conv7(x))
        return x


class CRNN(nn.Module):
    def __init__(self, n_classes=10, in_channels=3, use_rnn=True, cnn="base_CNN", feature_map_size=512, rnn_h=256):
        super(CRNN, self).__init__()
        self.use_rnn = use_rnn

        # cnn backbone
        if cnn == "base_CNN":
            self.cnn = base_CNN(in_channels=in_channels)
        elif cnn == "dpn":
            self.cnn = dpn.DPN92(in_channels=in_channels)
        
        # last layer use BiLSTM or Linear
        if self.use_rnn:
            self.rnn = nn.Sequential(
                BidirectionalLSTM(feature_map_size, rnn_h, rnn_h),
                BidirectionalLSTM(rnn_h, rnn_h, n_classes)
            )
        else:
            self.linear = nn.Linear(feature_map_size, n_classes)
    
    def forward(self, input_tensor):
        x = self.cnn(input_tensor)
        _, _, h, _ = x.shape # b, c, h, w
        assert h == 1, "the height of feature map should be 1"
        x = x.squeeze(2)

        if self.use_rnn:
            x = x.permute(2, 0, 1)
            x = self.rnn(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.linear(x)
            x = x.permute(1, 0, 2)
        return x



if __name__ == "__main__":
    crnn = CRNN(cnn="dpn", feature_map_size=2560, n_classes=63).to("cuda")
    input_tensor = torch.randn(10, 3, 32, 400).to("cuda")
    out = crnn(input_tensor)
    print(out.shape)
