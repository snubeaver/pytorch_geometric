import torch

from torch import autograd 
EPS = 1e-15
def arccosh(x):
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    return c0 + c1

def dense_diff_pool(x, adj, s, mask=None):
    r"""Differentiable pooling operator from the `"Hierarchical Graph
    Representation Learning with Differentiable Pooling"
    <https://arxiv.org/abs/1806.08804>`_ paper

    .. math::
        \mathbf{X}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{X}

        \mathbf{A}^{\prime} &= {\mathrm{softmax}(\mathbf{S})}^{\top} \cdot
        \mathbf{A} \cdot \mathrm{softmax}(\mathbf{S})

    based on dense learned assignments :math:`\mathbf{S} \in \mathbb{R}^{B
    \times N \times C}`.
    Returns pooled node feature matrix, coarsened adjacency matrix and the
    auxiliary link prediction objective :math:`\| \mathbf{A} -
    \mathrm{softmax}(\mathbf{S}) \cdot {\mathrm{softmax}(\mathbf{S})}^{\top}
    \|_F`.

    Args:
        x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
            \times N \times F}` with batch-size :math:`B`, (maximum)
            number of nodes :math:`N` for each graph, and feature dimension
            :math:`F`.
        adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
            \times N \times N}`.
        s (Tensor): Assignment tensor :math:`\mathbf{S} \in \mathbb{R}^{B
            \times N \times C}` with number of clusters :math:`C`. The softmax
            does not have to be applied beforehand, since it is executed
            within this method.
        mask (ByteTensor, optional): Mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
        :class:`Tensor`)
    """
    #print("in nn.dense_diff_pool")
    x = x.unsqueeze(0) if x.dim() == 2 else x
    adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
    s = s.unsqueeze(0) if s.dim() == 2 else s

    batch_size, num_nodes, _ = x.size()
    # (Batch x current node x next node)
    s = torch.softmax(s, dim=-1)
    inv_s = torch.diag_embed(torch.sum(s**s, dim=-1))
    degree = torch.diag_embed(adj.sum(dim=-1))
    lap = degree - adj
    import pdb

    new_lap = torch.matmul(inv_s.transpose(1,2), torch.matmul(lap, s))
    #pdb.set_trace()
    x_eig = autograd.Variable(lap[:,:,0:1])
    for i in range(lap.size(-1)):
        _, sec_eigen = lap[i,:,:].eig(eigenvectors=True) 
        x_eig[i,:,0] = sec_eigen[:,1]
    pdb.set_trace()
    spec_loss = arccosh(1 + torch.sum(torch.matmul((lap-new_lap), x_eig)**torch.matmul((lap-new_lap), x_eig))*torch.sum(x_eig**x_eig)
                            /(2 * torch.matmul(x_eig.transpose(1,2), torch.matmul(lap,x_eig)) * torch.matmul(x_eig.transpose(1,2), torch.matmul(new_lap,x_eig)) )
                        )

    if mask is not None:
        mask = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * mask, s * mask

    out = torch.matmul(s.transpose(1, 2), x)
    out_adj = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

    link_loss = adj - torch.matmul(s, s.transpose(1, 2))
    link_loss = torch.norm(link_loss, p=2)
    link_loss = link_loss / adj.numel()

    ent_loss = (-s * torch.log(s + EPS)).sum(dim=-1).mean()

    return out, out_adj, link_loss, spec_loss
