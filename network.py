import math
import torch
from torch import nn
from torch.nn import functional as F

class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, forget_bias=1.0, padding=0):
        super(ConvLSTM, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=out_channels + in_channels, out_channels=4 * out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.forget_bias = forget_bias

    def forward(self, inputs, states):
        if states is None:
            states = (torch.zeros([inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]], device=inputs.device),
                      torch.zeros([inputs.shape[0], self.out_channels, inputs.shape[2], inputs.shape[3]], device=inputs.device))
        if not isinstance(states, tuple):
            raise TypeError('states type is not right')

        c, h = states
        if not (len(c.shape) == 4 and len(h.shape) == 4 and len(inputs.shape) == 4):
            raise TypeError('')

        inputs_h = torch.cat((inputs, h), dim=1)
        i_j_f_o = self.conv(inputs_h)
        i, j, f, o = torch.split(i_j_f_o,  self.out_channels, dim=1)

        new_c = c * torch.sigmoid(f + self.forget_bias) + torch.sigmoid(i) * torch.tanh(j)
        new_h = torch.tanh(new_c) * torch.sigmoid(o)

        return new_h, (new_c, new_h)


class network(nn.Module):
    def __init__(self, opt=None, channels=3, height=64, width=64):
        super(network, self).__init__()
        lstm_size = [32, 32, 64, 64, 128, 64, 32]
        lstm_size = [l//2 for l in lstm_size]   # ligthen network
        self.channels = channels
        self.opt = opt
        self.height = height
        self.width = width

        # N * 3 * H * W -> N * 32 * H/2 * W/2
        self.enc0 = nn.Conv2d(in_channels=channels, out_channels=lstm_size[0], kernel_size=5, stride=2, padding=2)
        self.enc0_norm = nn.LayerNorm([lstm_size[0], self.height//2, self.width//2])
        # N * 32 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm1 = ConvLSTM(in_channels=lstm_size[0], out_channels=lstm_size[0], kernel_size=5, padding=2)
        self.lstm1_norm = nn.LayerNorm([lstm_size[0], self.height//2, self.width//2])
        # N * 32 * H/2 * W/2 -> N * 32 * H/2 * W/2
        self.lstm2 = ConvLSTM(in_channels=lstm_size[0], out_channels=lstm_size[1], kernel_size=5, padding=2)
        self.lstm2_norm = nn.LayerNorm([lstm_size[1], self.height//2, self.width//2])

        # N * 32 * H/4 * W/4 -> N * 32 * H/4 * W/4
        self.enc1 = nn.Conv2d(in_channels=lstm_size[1], out_channels=lstm_size[1], kernel_size=3, stride=2, padding=1)
        # N * 32 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm3 = ConvLSTM(in_channels=lstm_size[1], out_channels=lstm_size[2], kernel_size=5, padding=2)
        self.lstm3_norm = nn.LayerNorm([lstm_size[2], self.height//4, self.width//4])
        # N * 64 * H/4 * W/4 -> N * 64 * H/4 * W/4
        self.lstm4 = ConvLSTM(in_channels=lstm_size[2], out_channels=lstm_size[3], kernel_size=5, padding=2)
        self.lstm4_norm = nn.LayerNorm([lstm_size[3], self.height//4, self.width//4])
        # pass in state and action

        # N * 64 * H/4 * W/4 -> N * 64 * H/8 * W/8
        self.enc2 = nn.Conv2d(in_channels=lstm_size[3], out_channels=lstm_size[3], kernel_size=3, stride=2, padding=1)

        # N * (10+64) * H/8 * W/8 -> N * 64 * H/8 * W/8
        self.enc3 = nn.Conv2d(in_channels=lstm_size[3], out_channels=lstm_size[3], kernel_size=1, stride=1)
        # N * 64 * H/8 * W/8 -> N * 128 * H/8 * W/8
        self.lstm5 = ConvLSTM(in_channels=lstm_size[3], out_channels=lstm_size[4], kernel_size=5, padding=2)
        self.lstm5_norm = nn.LayerNorm([lstm_size[4], self.height//8, self.width//8])
        in_dim = int(lstm_size[4] * self.height * self.width / 64)
        self.feat = nn.Linear(in_dim, 100)
        self.classifer = nn.Linear(100, 9)

    def forward(self, images):
        '''
        :param inputs: T * N * C * H * W
        :param state: T * N * C
        :param action: T * N * C
        :return:
        '''

        lstm_state1, lstm_state2, lstm_state3, lstm_state4, lstm_state5 = None, None, None, None, None
        for image in images[:-1]:

            enc0 = self.enc0_norm(torch.relu(self.enc0(image)))

            lstm1, lstm_state1 = self.lstm1(enc0, lstm_state1)
            lstm1 = self.lstm1_norm(lstm1)

            lstm2, lstm_state2 = self.lstm2(lstm1, lstm_state2)
            lstm2 = self.lstm2_norm(lstm2)

            enc1 = torch.relu(self.enc1(lstm2))

            lstm3, lstm_state3 = self.lstm3(enc1, lstm_state3)
            lstm3 = self.lstm3_norm(lstm3)

            lstm4, lstm_state4 = self.lstm4(lstm3, lstm_state4)
            lstm4 = self.lstm4_norm(lstm4)

            enc2 = torch.relu(self.enc2(lstm4))

            enc3 = torch.relu(self.enc3(enc2))

            lstm5, lstm_state5 = self.lstm5(enc3, lstm_state5)
            lstm5 = self.lstm5_norm(lstm5)

        feats = lstm5.view(lstm5.shape[0], -1)
        feats = self.feat(feats)
        feats = self.classifer(feats)
        feats = torch.softmax(feats, dim=1)

        return feats


if __name__ == '__main__':
    net = network()
    net([torch.zeros(16, 3, 64, 64),torch.zeros(16, 3, 64, 64),torch.zeros(16, 3, 64, 64)])