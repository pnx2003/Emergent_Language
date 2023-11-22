import torch
import torch.nn as nn

from environment import get_rule


rule = get_rule(3,3)


def state2str(state):
    now_str = ""
    for s in state:
        now_str += str(s)
    return now_str


class OutsideStateModel(nn.Module):
    def __init__(self, output_dim, hidden_dim, state_dim, vocab_size, embed_size=64):
        super(OutsideStateModel, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc1 = nn.Linear(in_features=embed_size, out_features=hidden_dim, bias=True)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, bias=True)
        self.act = nn.Tanh()
        self.out_head = nn.Linear(in_features=1, out_features=state_dim, bias=True)

    def forward(self, input):
        # input (Batch_size, input_dim=1)
        embed = self.embed(input) # embed (Batch_size, input_dim, embed_size)
        embed = torch.sum(embed, dim=1, keepdim=False)
        hidden_state = self.act(self.fc1(embed))
        output = self.act(self.fc2(hidden_state))
        # print(f"output.shape = {output.shape}")
        # print(f"output[:,i].shape = {output[:, 0].shape}")
        output_dist = torch.empty(size=(input.shape[0], self.output_dim, self.state_dim))
        for i in range(self.output_dim):
            output_dist[:,i,:] = self.act(self.out_head(output[:, i].unsqueeze(1)))
        output_dist = torch.softmax(output_dist, dim=-1)
        return output_dist # (Batch_size, output_dim, state_dim)


class OutsideComModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size):
        super(OutsideComModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=True)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=vocab_size, bias=True)
        self.act = nn.Tanh()

    def forward(self, input):
        # input (Batch_size, input_dim)
        input = input.float()
        hidden_state = self.act(self.fc1(input))
        output = self.act(self.fc2(hidden_state))
        output_dist = torch.softmax(output, dim=-1)
        return output_dist


if __name__ == "__main__":
    state_test = OutsideStateModel(output_dim=3, hidden_dim=128, state_dim=4, vocab_size=64)
    symbol = torch.tensor([[1]], dtype=torch.int64)

    state = torch.argmax(state_test(symbol), dim=-1)

    # print(f"state.shape = {state.shape}")
    com_test = OutsideComModel(input_dim=3, hidden_dim=128, vocab_size=64)
    com = com_test(state)
    # print(f"com.shape = {com.shape}")
