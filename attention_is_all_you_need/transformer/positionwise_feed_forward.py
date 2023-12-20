from torch import nn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.l1 = nn.Linear(d_model,d_ff )
        self.l2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x