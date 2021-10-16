""" utils for simplex-net algorithm """

import torch
import torch.nn.functional as F


def icm(qlevels, d_p):
    """ returns ICM solution given update direction (d_p := grad) and simplex (qlevels)
    """
    icmp = view_w_as_u(d_p, qlevels) # convert to desired matrix format (N x d)
    icmpi = icmp.argmin(dim=1, keepdim=True)
    ids = torch.arange(qlevels, dtype=torch.long, device=icmpi.device)
    ids = ids.repeat(icmp.size(0), 1)
    icmp = ids.eq(icmpi)
    icmp = icmp.float()
    return view_u_as_w(icmp, d_p)


def mf(qlevels, d_p):
    """ returns MF solution given update direction (d_p := grad) and simplex (qlevels)
    """
    mfp = view_w_as_u(d_p, qlevels) # convert to desired matrix format (N x d)
    mfp = F.softmin(mfp, dim=1)
    return view_u_as_w(mfp, d_p)


def normalize(qlevels, p):
    """ returns normalized solution (sum to 1) given data and simplex (qlevels)
    """
    mfp = view_w_as_u(p, qlevels) # convert to desired matrix format (N x d)
    mfpsum = mfp.sum(dim=1).unsqueeze_(dim=1)
    mfp = mfp / mfpsum.expand_as(mfp)
    return view_u_as_w(mfp, p)


def normalize_stable(qlevels, p_data, exponential_term):
    """ returns normalized solution (sum to 1) given data and simplex (qlevels)
    """
    mf_pdata = view_w_as_u(p_data, qlevels) # convert to desired matrix format (N x d)
    mf_expterm = view_w_as_u(exponential_term, qlevels) # convert to desired matrix format (N x d)
    mf_expterm = torch.exp(mf_expterm - torch.max(mf_expterm, dim=1)[0].unsqueeze(1).repeat(1, 2)) # for stability we subtract exponentional term with max(exponential term), to avoid Nan problems due to overflow of exp term (motivated from softmax)

    epsilon = 1e-20    # avoid divisin by 0 and for a j, \sum_{lambda} u_{lambda} = 1, so all u_{lambda} cannot be 0
    mfp = mf_pdata * (mf_expterm + epsilon)
    # mfp = mfp + epsilon # if [0,0] --> [0.5, 0.5]

    mfpsum = mfp.sum(dim=1).unsqueeze_(dim=1)
    mfp = mfp / mfpsum.expand_as(mfp)
    if torch.isnan(mfp).any():
        print 'NaN encountered in mfp'
    return view_u_as_w(mfp, p_data)


def isfeasible(qlevels, wo):
    """ checks if w is in the polytope
    """
    epsilon = 1e-5    # for floating point errors
    w = view_w_as_u(wo, qlevels) # convert to desired matrix format (N x d)
    tmp = w.lt(0. - epsilon).byte()
    if tmp.any():
        return False
    w = w.sum(dim=1)
    tmp1 = w.gt(1. + epsilon).byte()
    tmp2 = w.lt(1. - epsilon).byte()
    if tmp1.any() or tmp2.any():
        return False
    return True


def normalize_unit(u):
    """ normalize to be in [0,1]
    """
    w = u - u.min()
    w.div_(w.max())
    return w


def sparsemax(u, qlevels):
    """ sparsemax (Euclidean) projection
    """
    ws, wi = u.sort(dim=1, descending=True)
    ind = torch.arange(qlevels, dtype=torch.long, device=u.device) + 1
    cssv = ws.cumsum(dim=1) - 1
    cond = ws - cssv.div(ind.float()) > 0
    rho1 = torch.where(cond, ind, torch.zeros(ind.size(), dtype=torch.long, device=u.device))
    rho, ids = rho1.max(dim=1, keepdim=True)
    cond1 = rho1 - rho == 0
    neginf = torch.zeros(cssv.size(), device=u.device)
    neginf.fill_(-float('Inf'))
    crho = torch.where(cond1, cssv, neginf)
    crho, ids = crho.max(dim=1, keepdim=True)
    theta = crho.div(rho.float())
    w = u - theta
    w.clamp_(0., 1.)
    return w


def view_w_as_u(w, d):
    """ view given w (dN1 x N2) as u (N1N2 x d)
    """
    szl = len(w.size())
    if szl == 1:   # bias   (dN --> N x d)
        u = w.view(-1, d)
    elif szl == 2:  # linear weights    (dN1 x N2 --> N1N2 x d)
        u = w.permute(1,0).contiguous().view(-1, d) # permute(1,0) == transpose(0,1)
    elif szl == 4:  # conv2d weights    (dN1 x N2 x N3 x N4 --> N1N2N3N4 x d)
        u = w.permute(1, 2, 3, 0).contiguous().view(-1, d)
    else:
        print 'Weight size "{0}" not recognized, exiting ...'.format(w.size())
        exit()
    return u


def view_u_as_w(u, wo):
    """ view given u (N1N2 x d) as wo (dN1 x N2) --> reverse of "view_w_as_u"
    """
    szl = len(wo.size())
    if szl == 1:   # bias   (N x d --> dN)
        w = u.view_as(wo)
    elif szl == 2:  # linear weights    (N1N2 x d --> dN1 x N2)
        w = u.view_as(wo.permute(1,0)).permute(1,0)
    elif szl == 4:  # conv2d weights    (N1N2N3N4 x d --> dN1 x N2 x N3 x N4)
        w = u.view_as(wo.permute(1, 2, 3, 0)).permute(3, 0, 1, 2)
    else:
        print 'Weight size "{0}" not recognized, exiting ...'.format(wo.size())
        exit()
    return w