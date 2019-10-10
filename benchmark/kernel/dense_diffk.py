import torch
from math import ceil, inf

EPS = 1e-15


def dense_diffk(x, adj, s, mask=None):
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    #print(x.size())
    #print(s.size())
    batch_size, num_nodes, _ = x.size()

    _, in_node, out_node = s.size()
    #s.view(-1, out_node)
    topk, inds = torch.topk(s, ceil(in_node*0.4), dim = 1)[1]
    s= s.view(-1, out_node)
    inds = inds.view(-1, out_node)
    res = Variable(torch.zeros(s.size())
    res = res.scatter(0, inds, 1)
    s = s.view(batch_size, -1, out_node)
    res = res.view(batch_size, -1, out_node)
    s = torch.mul(s,res)
    #print(">>>>>>>>>>>>>>>>")
    #print(s.size())
    #print(inds.size())
    s = torch.softmax(s, dim=-1)

    if mask is not None:

        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

    return out, out_adj, link_loss, ent_loss
