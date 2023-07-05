import torch
import random


def t2i_sim(sim_matrix, k=3):
    if sim_matrix.shape[0] == 0:
        return torch.zeros(1, dtype=sim_matrix.dtype, device=sim_matrix.device).squeeze()
    # f_sim = sim_matrix.max(dim=1)[0]
    target_k = min(k, sim_matrix.shape[1])
    f_sim = sim_matrix.topk(target_k, dim=1)[0]
    rand_index = torch.randint(0, target_k, (f_sim.shape[0],), device=f_sim.device)
    f_sim = f_sim[torch.arange(f_sim.shape[0], device=f_sim.device), rand_index]
    return torch.mean(f_sim)

def t2i_sim_max(sim_matrix):
    if sim_matrix.shape[0] == 0:
        return torch.zeros(1, dtype=sim_matrix.dtype, device=sim_matrix.device).squeeze()
    f_sim = sim_matrix.max(dim=1)[0]
    # f_sim = sim_matrix.topk(3, dim=1)[0]
    # rand_index = torch.randint(0, 3, (f_sim.shape[0],), device=f_sim.device)
    # f_sim = f_sim[torch.arange(f_sim.shape[0], device=f_sim.device), rand_index]
    return torch.mean(f_sim)

def i2t_sim(sim_matrix, k=3):
    if sim_matrix.shape[1] == 0:
        return torch.zeros(1, dtype=sim_matrix.dtype, device=sim_matrix.device).squeeze()
    # f_sim = sim_matrix.max(dim=1)[0]
    target_k = min(k, sim_matrix.shape[0])
    f_sim = sim_matrix.topk(target_k, dim=0)[0]
    rand_index = torch.randint(0, target_k, (f_sim.shape[1],), device=f_sim.device)
    f_sim = f_sim[rand_index, torch.arange(f_sim.shape[1], device=f_sim.device)]
    return torch.mean(f_sim)

def i2t_sim_max(sim_matrix):
    if sim_matrix.shape[1] == 0:
        return torch.zeros(1, dtype=sim_matrix.dtype, device=sim_matrix.device).squeeze()
    f_sim = sim_matrix.max(dim=0)[0]
    # f_sim = sim_matrix.topk(3, dim=1)[0]
    # rand_index = torch.randint(0, 3, (f_sim.shape[0],), device=f_sim.device)
    # f_sim = f_sim[torch.arange(f_sim.shape[0], device=f_sim.device), rand_index]
    return torch.mean(f_sim)


def get_pos_neg_sims(sims, src_mask, tar_mask, hard_index=None, sim_method='max', k=3):
    src_n_input = torch.sum(src_mask, dim=-1)
    tar_n_input = torch.sum(tar_mask, dim=-1)
    src_index_border = src_n_input.cumsum(dim=0)
    tar_index_border = tar_n_input.cumsum(dim=0)
    my_zero = torch.zeros(1, device=sims.device, dtype=src_index_border.dtype)

    src_index_border = torch.cat([my_zero, src_index_border], dim=0)
    tar_index_border = torch.cat([my_zero, tar_index_border], dim=0)

    def tmp_sim(sim_mat):
        if sim_method == 'max':
            return t2i_sim_max(sim_mat)
        elif sim_method == 'topk':
            return t2i_sim(sim_mat, k=k)
        else:
            raise NotImplementedError
    
    def tmp_sim_reverse(sim_mat):
        if sim_method == 'max':
            return i2t_sim_max(sim_mat)
        elif sim_method == 'topk':
            return i2t_sim(sim_mat, k=k)
        else:
            raise NotImplementedError

    doc2pos_sim = []
    doc2pos_sim_re = []
    doc2neg_img_sims = []
    doc2neg_img_sims_re = []

    for src_idx in range(src_mask.shape[0]):
        # doc2pos_sim[src_idx] = tmp_sim(sims[src_index_border[src_idx]:src_index_border[src_idx+1], tar_index_border[src_idx]:tar_index_border[src_idx+1]])
        src_start = src_index_border[src_idx]
        src_end = src_index_border[src_idx+1]
        tar_start = tar_index_border[src_idx]
        tar_end = tar_index_border[src_idx+1]
        doc2pos_sim.append(tmp_sim(sims[src_start:src_end, tar_start:tar_end]))
        if hard_index is None:
            neg_tar_indexs = list(range(0, src_idx)) + list(range(src_idx + 1, tar_mask.shape[0]))
            neg_tar_idx = random.choice(neg_tar_indexs)
        else:
            neg_tar_idx = hard_index[src_idx]
        neg_tar_start = tar_index_border[neg_tar_idx]
        neg_tar_end = tar_index_border[neg_tar_idx+1]
        doc2neg_img_sims.append(tmp_sim(sims[src_start:src_end, neg_tar_start:neg_tar_end]))
        # for img_idx in range(img_index.shape[0]):
        #     text_start = text_index_border[text_idx]
        #     text_end = text_index_border[text_idx+1]
        #     img_start = img_index_border[img_idx]
        #     img_end = img_index_border[img_idx+1]
        #     c_sim = t2i_sim(sims[text_start:text_end, img_start:img_end])
        #     if text_idx == img_idx:
        #         doc2pos_sim[text_idx] = c_sim
        #     else:
        #         doc2neg_img_sims[text_idx].append(c_sim)

    pos_sims = torch.stack(doc2pos_sim)
    neg_sims = torch.stack(doc2neg_img_sims)

    return pos_sims, neg_sims


def get_sims_from_mats_s2t(sims, src_mask, tar_mask, bi_direction=False, sim_method='max', k=3):
    # get the sim from the sim matrix by MIL assumption
    src_n_input = torch.sum(src_mask, dim=-1)
    tar_n_input = torch.sum(tar_mask, dim=-1)

    def tmp_sim(sim_mat):
        if sim_method == 'max':
            return t2i_sim_max(sim_mat)
        elif sim_method == 'topk':
            return t2i_sim(sim_mat, k=k)
        else:
            raise NotImplementedError
    
    def tmp_sim_reverse(sim_mat):
        if sim_method == 'max':
            return i2t_sim_max(sim_mat)
        elif sim_method == 'topk':
            return i2t_sim(sim_mat, k=k)
        else:
            raise NotImplementedError

    doc2pos_sim = []
    if bi_direction:
        # bi-directional
        doc2pos_sim_re = []

    for src_idx in range(src_mask.shape[0]):
        # doc2pos_sim[src_idx] = tmp_sim(sims[src_index_border[src_idx]:src_index_border[src_idx+1], tar_index_border[src_idx]:tar_index_border[src_idx+1]])
        src_end = src_n_input[src_idx]
        tar_end = tar_n_input[src_idx]
        tmp_mat = sims[src_idx, :src_end, :tar_end]
        doc2pos_sim.append(tmp_sim(tmp_mat))
        if bi_direction:
            doc2pos_sim_re.append(tmp_sim_reverse(tmp_mat))

    pos_sims = torch.stack(doc2pos_sim)
    if bi_direction:
        pos_sims_re = torch.stack(doc2pos_sim_re)
        return pos_sims, pos_sims_re

    return pos_sims

def get_sims_from_mats_t2s(sims, src_mask, tar_mask, sim_method='max', k=3):
    # get the sim from the sim matrix by MIL assumption
    src_n_input = torch.sum(src_mask, dim=-1)
    tar_n_input = torch.sum(tar_mask, dim=-1)

    def tmp_sim(sim_mat):
        if sim_method == 'max':
            return t2i_sim_max(sim_mat)
        elif sim_method == 'topk':
            return t2i_sim(sim_mat, k=k)
        else:
            raise NotImplementedError
    
    def tmp_sim_reverse(sim_mat):
        if sim_method == 'max':
            return i2t_sim_max(sim_mat)
        elif sim_method == 'topk':
            return i2t_sim(sim_mat, k=k)
        else:
            raise NotImplementedError

    doc2pos_sim = []

    for src_idx in range(src_mask.shape[0]):
        # doc2pos_sim[src_idx] = tmp_sim(sims[src_index_border[src_idx]:src_index_border[src_idx+1], tar_index_border[src_idx]:tar_index_border[src_idx+1]])
        src_end = src_n_input[src_idx]
        tar_end = tar_n_input[src_idx]
        doc2pos_sim.append(tmp_sim_reverse(sims[src_idx, :src_end, :tar_end]))

    pos_sims = torch.stack(doc2pos_sim)

    return pos_sims