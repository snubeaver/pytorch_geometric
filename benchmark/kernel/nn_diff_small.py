import torch

from torch import autograd 
EPS = 1e-15
def arccosh(x):
    c0 = torch.log(x)
    #print(c0)
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
    s = torch.softmax(s, dim=-1)  # 28 x 5
    diag_s = torch.diag_embed(torch.sum(s*s, dim=-1).pow(-2))   # 28 x 28
    inv_s = torch.matmul(s.transpose(1,2) , diag_s) # 5 x  28
    new_adj = torch.matmul(s.transpose(1,2), torch.matmul(adj, s) )
    degree =  torch.diag_embed(adj.sum(dim=-1)) # 28 x 28
    new_degree = torch.diag_embed(new_adj.sum(dim=-1)) # 28 x 28
    o_lap = new_degree - new_adj # 5 x 5
    lap = degree - adj
    

    new_lap = torch.matmul(inv_s.transpose(1,2), torch.matmul(o_lap, inv_s)) # 28 x 5
    # x_eig = autograd.Variable(lap[:,:,0:1])
    import pdb
	#pdb.set_trace()
    u_, s_, v_ = torch.svd(lap)
    zeros = torch.zeros(s_.size()).cuda()
    new_s= torch.where(s_<=1,zeros, s_)
    indices = torch.argmin(new_s, dim=1)
    x_eig = torch.index_select(v_, 1, indices)
	# output[i][j][k] = x[i][indice[i][j][:]
	#x_eig = v_.[:,0,:].detach()
    pdb.set_trace()
    x_eig = x_eig.unsqueeze(2)
    '''
    x_eig = torch.zeros(*lap[:,:,0:1].shape).to('cuda:0')
    import pdb
    
    for i in range(lap.size(0)):
        value, sec_eigen = lap[i,:,:].eig(eigenvectors=True)
        nonzero = (value[:,0]<=0).sum(0)
        sortedvalue, indice = value.sort(dim=0)
        index = indice[nonzero,0]
        #pdb.set_trace()
        x_eig[i,:,0] = sec_eigen[:,index].detach()
    '''
    
    lapnorm = torch.norm(torch.matmul((lap-new_lap), x_eig),p=2)
    xnorm = torch.norm(torch.matmul(x_eig.transpose(1,2), x_eig), p=2)
    spec_loss = arccosh(1 + 
    lapnorm*lapnorm + xnorm*xnorm*2
    /(2 * torch.sum(torch.matmul( torch.matmul(x_eig.transpose(1,2), torch.matmul(lap,x_eig)) , torch.matmul(x_eig.transpose(1,2), torch.matmul(new_lap,x_eig)) ))
    ))
    #print(torch.matmul((lap-new_lap), x_eig))
    #print(torch.matmul(x_eig.transpose(1,2), torch.matmul(lap,x_eig)))
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
