import os
import logging
import numpy as np
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import BatchType, TestDataset


class KGEModel(nn.Module, ABC):
    """
    Must define
        `self.entity_embedding`
        `self.relation_embedding`
    in the subclasses.
    """

    @abstractmethod
    def func(self, head, rel, tail, batch_type):
        """
        Different tensor shape for different batch types.
        BatchType.SINGLE:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.HEAD_BATCH:
            head: [batch_size, negative_sample_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, hidden_dim]

        BatchType.TAIL_BATCH:
            head: [batch_size, hidden_dim]
            relation: [batch_size, hidden_dim]
            tail: [batch_size, negative_sample_size, hidden_dim]
        """
        ...

    def forward(self, sample, batch_type=BatchType.SINGLE):
        """
        Given the indexes in `sample`, extract the corresponding embeddings,
        and call func().

        Args:
            batch_type: {SINGLE, HEAD_BATCH, TAIL_BATCH},
                - SINGLE: positive samples in training, and all samples in validation / testing,
                - HEAD_BATCH: (?, r, t) tasks in training,
                - TAIL_BATCH: (h, r, ?) tasks in training.

            sample: different format for different batch types.
                - SINGLE: tensor with shape [batch_size, 3]
                - {HEAD_BATCH, TAIL_BATCH}: (positive_sample, negative_sample)
                    - positive_sample: tensor with shape [batch_size, 3]
                    - negative_sample: tensor with shape [batch_size, negative_sample_size]
        """
        # sample = torch.tensor(sample, dtype=torch.long)
        if batch_type == BatchType.SINGLE:
            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.HEAD_BATCH:
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head_part = head_part.clone().detach().long()
            tail_part = tail_part.clone().detach().long()

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif batch_type == BatchType.TAIL_BATCH:
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head_part = head_part.clone().detach().long()
            tail_part = tail_part.clone().detach().long()

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('batch_type %s not supported!'.format(batch_type))

        # return scores
        return self.func(head, relation, tail, batch_type)

    @staticmethod
    def train_step(model, optimizer, train_iterator, args, use_cuda=False):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, batch_type = next(train_iterator)

        if use_cuda == True:
            positive_sample = positive_sample.cuda()
        if use_cuda == True:
            negative_sample = negative_sample.cuda()
        if use_cuda == True:
            subsampling_weight = subsampling_weight.cuda()

        # negative scores
        negative_score = model((positive_sample, negative_sample), batch_type=batch_type)

        negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                          * F.logsigmoid(-negative_score)).sum(dim=1)

        # positive scores
        positive_score = model(positive_sample)

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

        positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
        negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        loss.backward()

        optimizer.step()

        log = {
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, data_reader, mode, args, use_cuda=False):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        test_dataloader_head = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.HEAD_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                data_reader,
                mode,
                BatchType.TAIL_BATCH
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, batch_type in test_dataset:
                    if use_cuda == True:
                        positive_sample = positive_sample.cuda()
                    if use_cuda == True:
                        negative_sample = negative_sample.cuda()
                    if use_cuda == True:
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score = model((positive_sample, negative_sample), batch_type)
                    score = score + filter_bias

                    # explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if batch_type == BatchType.HEAD_BATCH:
                        positive_arg = positive_sample[:, 0]
                    elif batch_type == BatchType.TAIL_BATCH:
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... ({}/{})'.format(step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, batch_type=None, scale=None, attn_mask=None):
        """Forward function

        Args:
        	q: Queries tensor with shape of [B, L_q, D_q]
        	k: Keys tensor with shape of [B, L_k, D_k]
        	v: Values tensor with shape of [B, L_v, D_v], which usually is k
        	scale: Zoom factor, which usually is a float scalar
        	attn_mask: Masking tensor with shape of [B, L_q, L_k]

        Returns:
        	Context tensor and attetention tensor
        """
        if batch_type == BatchType.TAIL_BATCH:
            attention = torch.bmm(q, k.transpose(1, 2))
            if scale:
                attention = attention * scale
            if attn_mask:
                # set a negative infinity where mask is required
                attention = attention.masked_fill_(attn_mask, -np.inf)
            # calculate softmax
            attention = self.softmax(attention)
            # add dropout
            attention = self.dropout(attention)

            # logging.info('shape q:{} k:{} v:{} transK:{} att:{}'.format(
            #     q.shape, k.shape, v.shape,k.transpose(1, 2).shape, attention.shape))
            # get dot product with V
            context = torch.bmm(attention, v)
            return context, attention
        else:
            attention = torch.bmm(q, k.transpose(1, 2))
            if scale:
                attention = attention * scale
            if attn_mask:
                attention = attention.masked_fill_(attn_mask, -np.inf)
            attention = self.softmax(attention)
            attention = self.dropout(attention)

            context = torch.bmm(attention.transpose(1, 2), v)
            return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # layer norm is needed after calculating multi-head attention
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, batch_type=None, attn_mask=None):
		# residual Connection
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = 1
        # scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, batch_type, scale, attn_mask)
        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        # final linear projection
        output = self.linear_final(context)
        # dropout
        output = self.dropout(output)
        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention



class ModE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma):
        super(ModE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

    def func(self, head, rel, tail, batch_type):
        return self.gamma.item() - torch.norm(head * rel - tail, p=1, dim=2)


class HAKE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma, modulus_weight=1.0, phase_weight=0.5):
        super(HAKE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_embedding = nn.Parameter(torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(num_relation, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_(
            tensor=self.relation_embedding[:, hidden_dim:2 * hidden_dim]
        )

        nn.init.zeros_(
            tensor=self.relation_embedding[:, 2 * hidden_dim:3 * hidden_dim]
        )

        self.phase_weight = nn.Parameter(torch.Tensor([[phase_weight * self.embedding_range.item()]]))
        self.modulus_weight = nn.Parameter(torch.Tensor([[modulus_weight]]))

        self.pi = 3.14159265358979323846


    def func(self, head, rel, tail, batch_type):
        phase_head, mod_head = torch.chunk(head, 2, dim=2)
        phase_relation, mod_relation, bias_relation = torch.chunk(rel, 3, dim=2)
        phase_tail, mod_tail = torch.chunk(tail, 2, dim=2)

        phase_head = phase_head / (self.embedding_range.item() / self.pi)
        phase_relation = phase_relation / (self.embedding_range.item() / self.pi)
        phase_tail = phase_tail / (self.embedding_range.item() / self.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = phase_head + (phase_relation - phase_tail)
        else:
            phase_score = (phase_head + phase_relation) - phase_tail

        mod_relation = torch.abs(mod_relation)
        bias_relation = torch.clamp(bias_relation, max=1)
        indicator = (bias_relation < -mod_relation)
        bias_relation[indicator] = -mod_relation[indicator]

        r_score = mod_head * (mod_relation + bias_relation) - mod_tail * (1 - bias_relation)

        phase_score = torch.sum(torch.abs(torch.sin(phase_score / 2)), dim=2) * self.phase_weight
        r_score = torch.norm(r_score, dim=2) * self.modulus_weight

        return self.gamma.item() - ( phase_score + r_score)
        

class CirE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma):
        super(CirE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        # self.embedding_range = nn.Parameter(
        #     torch.Tensor([6/torch.sqrt(torch.tensor(hidden_dim).float())]),
        #     requires_grad=False
        # )

        self.entity_embedding = nn.Parameter(
            torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(
            torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
    
    def poin_dis(self, u ,v):
        # logging.info('shape u:{} \t v :{}'.format(u.shape, v.shape))
        delta = 2 * torch.norm(u-v, p=2, dim=2) / ((1-torch.norm(u, p=2, dim=2)) * (1-torch.norm(v, p=2, dim=2)))
        dis = torch.acosh(1+delta)
        # dis = torch.clamp(torch.acosh(1+delta), max=3, min=2)
        # logging.info(
        #     'Delta / Dis ({}/{})'.format(u.detach().numpy(), v.detach().numpy()))
        return dis

    def func(self, head, rel, tail, batch_type):
        if batch_type == BatchType.HEAD_BATCH:
            score = head + (rel - tail)
        else:
            score = (head + rel) - tail

        return self.gamma.item() - torch.norm(torch.cosh(score), p=1, dim=2)
        
        # head = head / torch.norm(head, p=2)
        # tail = tail / torch.norm(tail, p=2)
        # # return self.gamma.item() - self.poin_dis(head + rel, tail)
        # return self.gamma.item() - torch.norm(head + rel - tail, p=2, dim=2)
        # logging.info('gamma :{}'.format(self.gamma.item()))
        # logging.info('shape head:{} \t rel:{} \t tail :{}'.format(head.shape, rel.shape, tail.shape))
        # return self.gamma.item() - (torch.norm(head - rel, p=1, dim=2) + torch.norm(tail - rel, p=1, dim=2) - torch.norm(head - tail, p=1, dim=2))


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def tanh(x):
    return x.clamp(-15, 15).tanh()


class HypE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma):
        super(HypE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        # self.embedding_range = nn.Parameter(
        #     torch.Tensor([1]),
        #     requires_grad=False
        # )

        self.entity_embedding = nn.Parameter(
            torch.zeros(num_entity, hidden_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # self.entity_embedding = nn.Parameter(self.entity_embedding /
        #                                      torch.sqrt((self.entity_embedding * self.entity_embedding).sum(-1,keepdim=True)))

        self.relation_embedding = nn.Parameter(
            torch.zeros(num_relation, hidden_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        self.curv = nn.Parameter(torch.tensor([1.0]), requires_grad=True)

        # self.attention_e = ScaledDotProductAttention()
        # self.attention_e = MultiHeadAttention(model_dim=self.hidden_dim, num_heads=16)
        # self.relation_embedding = nn.Parameter(self.relation_embedding /
        #                                      torch.sqrt((self.relation_embedding * self.relation_embedding).sum(-1,keepdim=True)))
        self.MIN_NORM = 1e-15
        self.MAX_NORM = 15
        self.BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


    def poin_dis(self, u, v):
        # logging.info('shape u:{} \t v :{}'.format(u.shape, v.shape))

        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= self.MAX_NORM,
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-self.MIN_NORM), u)
        v = torch.where(torch.norm(v, 2, dim=-1, keepdim=True) >= self.MAX_NORM,
                        v/(torch.norm(v, 2, dim=-1, keepdim=True)-self.MIN_NORM), v)

        delta_uv = torch.where(torch.norm(u-v, 2, dim=-1, keepdim=True) >= self.MAX_NORM,
                        (u-v)/(torch.norm(u-v, 2, dim=-1, keepdim=True)-1e-5), u-v)

        delta = 2 * torch.norm(delta_uv, p=2, dim=2, keepdim=True)**2 / \
            ((1-torch.norm(u, p=2, dim=2, keepdim=True)**2 + self.MIN_NORM) *
             (1-torch.norm(v, p=2, dim=2, keepdim=True)**2 + self.MIN_NORM))
        dis = torch.acosh(1+delta)
        # dis = 1+delta

        # dis = torch.clamp(torch.acosh(1+delta), max=3, min=2)
        # logging.info(
        #     'Delta / Dis ({}/{})'.format(u.detach().numpy(), v.detach().numpy()))
        return dis
    
    def poin_norm2(self, u):
        u = torch.where(torch.norm(u, 2, dim=-1, keepdim=True) >= self.MAX_NORM,
                        u/(torch.norm(u, 2, dim=-1, keepdim=True)-self.MIN_NORM), u)
        tmp = 2 * torch.norm(u, p=2, dim=2, keepdim=True)/ \
            ((1-torch.norm(u, p=2, dim=2, keepdim=True)**2 + self.MIN_NORM))
        dis = torch.acosh(1+tmp)
        return dis

    def project(self, x, c=1):
        """Project points to Poincare ball with curvature c.

        Args:
            x: torch.Tensor of size B x d with hyperbolic points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

        Returns:
            torch.Tensor with projected hyperbolic points.
        """
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        eps = self.BALL_EPS[x.dtype]
        maxnorm = (1 - eps) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)
    
    def extend_project(self, x, c=1):
        """Project points to Poincare ball with curvature c.

        Args:
            x: torch.Tensor of size B x d with hyperbolic points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

        Returns:
            torch.Tensor with projected hyperbolic points.
        """
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        eps = self.BALL_EPS[x.dtype]
        maxnorm = (1 - eps) / (c ** 0.5)
        cond = norm > maxnorm
        projected = (x / norm * maxnorm)
        result = artanh(torch.where(cond, projected, x) )
        return result.clamp_max(15).clamp_min(self.MIN_NORM)


    def mobius_add(self, x, y, c=1):
        """Mobius addition of points in the Poincare ball with curvature c.

        Args:
            x: torch.Tensor of size B x d with hyperbolic points
            y: torch.Tensor of size B x d with hyperbolic points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

        Returns:
            Tensor of shape B x d representing the element-wise Mobius addition of x and y.
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True).clamp_max(self.MAX_NORM)
        y2 = torch.sum(y * y, dim=-1, keepdim=True).clamp_max(self.MAX_NORM)
        xy = torch.sum(x * y, dim=-1, keepdim=True).clamp_max(self.MAX_NORM)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.MIN_NORM)


    # ################# HYP DISTANCES ########################

    def hyp_distance(self, x, y, c=1, eval_mode=False):
        """Hyperbolic distance on the Poincare ball with curvature c.

        Args:
            x: torch.Tensor of size B x d with hyperbolic queries
            y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)
            c: torch.Tensor of size 1 with absolute hyperbolic curvature

        Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
                else B x n_entities matrix with all pairs distances
        """
        sqrt_c = c ** 0.5
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        if eval_mode:
            y2 = torch.sum(y * y, dim=-1, keepdim=True).transpose(0, 1)
            xy = x @ y.transpose(0, 1)
        else:
            y2 = torch.sum(y * y, dim=-1, keepdim=True)
            xy = torch.sum(x * y, dim=-1, keepdim=True)
        c1 = 1 - 2 * c * xy + c * y2
        c2 = 1 - c * x2
        num = torch.sqrt((c1 ** 2) * x2 + (c2 ** 2) * y2 - (2 * c1 * c2) * xy)
        denom = 1 - 2 * c * xy + c ** 2 * x2 * y2
        pairwise_norm = num / denom.clamp_min(self.MIN_NORM)
        dist = artanh(sqrt_c * pairwise_norm)
        return 2 * dist / sqrt_c

    def sim_inner_dot(self, u ,v):
        """
        Retruns: hyperbolic inner product
        """
        a = self.poin_norm2(u)
        b = self.poin_norm2(v)
        c = self.poin_dis(u, v)
        result = a*b* (torch.cosh(a).clamp_max(self.MAX_NORM)*torch.cosh(b).clamp_max(self.MAX_NORM)-torch.cosh(c).clamp_max(self.MAX_NORM))
        result = result / (torch.sinh(a)*torch.sinh(b)).clamp_min(self.MIN_NORM)
        return result

    def minkowski_dot(self, u, v):
        tmp_u = torch.zeros(u.shape).cuda()
        tmp_u[:,:,0] = u[:,:,0]
        tmp_v = torch.zeros(v.shape).cuda()
        tmp_v[:, :, 0] = v[:, :, 0]
        dis = 2 * tmp_u * tmp_v - u*v
        return dis

    def func(self, head, rel, tail, batch_type):
        
        # if batch_type == BatchType.HEAD_BATCH:
        #     score = (head + (rel - tail)) 
        # else:
        #     score = ((head + rel) - tail) 

        # return self.gamma.item() - torch.norm(score, p=1, dim=2)

        # head = head / torch.norm(head, p=2)
        # tail = tail / torch.norm(tail, p=2)
        # logging.info('shape head:{} tail:{} rel:{} dis:{} score:{} res:{}'.format(
        #     head.shape, tail.shape, rel.shape, self.poin_dis(head, tail).shape, score.shape, 233))
        # return self.gamma.item() - (self.poin_dis(head, tail) + torch.sum(score, dim=2))
        # return self.gamma.item() - torch.norm(head + rel - tail, p=2, dim=2)
        # logging.info('shape head:{} \t rel:{} \t tail :{}'.format(
        #     head.shape, rel.shape, tail.shape))

        # tmp = self.project(self.mobius_add(head, rel, c=1))


        # tmp = self.mobius_add(self.extend_project(head, self.curv),self.extend_project(rel, self.curv), self.curv)
        # result = self.sim_inner_dot(tmp, self.extend_project(tail, self.curv), self.curv)

        tmp = self.mobius_add(self.extend_project(
            head), self.extend_project(rel))
        result = self.sim_inner_dot(
            tmp, self.extend_project(tail))
       
        return self.gamma.item() - torch.norm(result, p=1, dim=2)
        # logging.info('gamma :{}'.format(self.gamma.item()))
        # logging.info('shape head:{} \t rel:{} \t tail :{}'.format(head.shape, rel.shape, tail.shape))
        # return self.gamma.item() - (torch.norm(head - rel, p=1, dim=2) + torch.norm(tail - rel, p=1, dim=2) - torch.norm(head - tail, p=1, dim=2))


class HypBallE(KGEModel):
    def __init__(self, num_entity, num_relation, hidden_dim, gamma):
        super(HypBallE, self).__init__()
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )
        # self.entity_embedding_range = nn.Parameter(
        #     torch.Tensor([1]),
        #     requires_grad=False
        # )

        self.entity_embedding = nn.Parameter(
            torch.zeros(num_entity, hidden_dim * 2))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # nn.init.xavier_normal_( tensor=self.entity_embedding[:,hidden_dim : hidden_dim*2] )

        self.relation_embedding = nn.Parameter(
            torch.zeros(num_relation, hidden_dim * 3))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        nn.init.ones_( tensor=self.relation_embedding[:, hidden_dim: hidden_dim*2] )

        nn.init.zeros_( tensor=self.relation_embedding[:, hidden_dim*2:hidden_dim*3] )

        self.project_embedding = nn.Parameter(
            torch.zeros(hidden_dim, hidden_dim))
        nn.init.xavier_normal_(
            tensor=self.project_embedding
        )
        self.pi = 3.141592653589793238463
        self.MIN_NORM = 1e-12
        self.BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}
        self.curv = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.p_weight = nn.Parameter(torch.tensor([[0.5 * self.embedding_range.item()]]))
        self.m_weight = nn.Parameter(torch.tensor([[0.005]]))
        self.h_weight = nn.Parameter(torch.tensor([[0.5]]))

    def poin_dis(self, u, v):
        # logging.info('shape u:{} \t v :{}'.format(u.shape, v.shape))
        delta = 2 * torch.norm(u-v, p=2, dim=2)**2 / \
            ((1-torch.norm(u, p=2, dim=2)**2).clamp_min(self.MIN_NORM)
             * (1-torch.norm(v, p=2, dim=2)**2 ).clamp_min(self.MIN_NORM))
        # dis = torch.acosh(1+delta)
        dis = 1+delta
        return dis

    def expmap0(self, u, c=1):
        """Exponential map taken at the origin of the Poincare ball with curvature c.

        Args:
            u: torch.Tensor of size B x d with hyperbolic points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

        Returns:
            torch.Tensor with tangent points.
        """
        sqrt_c = c ** 0.5
        u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
        return self.project(gamma_1, c)


    def logmap0(self, y, c=1):
        """Logarithmic map taken at the origin of the Poincare ball with curvature c.

        Args:
            y: torch.Tensor of size B x d with tangent points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

        Returns:
            torch.Tensor with hyperbolic points.
        """
        sqrt_c = c ** 0.5
        y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        return y / y_norm / sqrt_c * artanh(sqrt_c * y_norm)


    def project(self, x, c=1):
        """Project points to Poincare ball with curvature c.

        Args:
            x: torch.Tensor of size B x d with hyperbolic points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

        Returns:
            torch.Tensor with projected hyperbolic points.
        """
        norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(self.MIN_NORM)
        eps = self.BALL_EPS[x.dtype]
        maxnorm = (1 - eps) / (c ** 0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def mobius_add(self, x, y, c=1):
        """Mobius addition of points in the Poincare ball with curvature c.

        Args:
            x: torch.Tensor of size B x d with hyperbolic points
            y: torch.Tensor of size B x d with hyperbolic points
            c: torch.Tensor of size 1 or B x 1 with absolute hyperbolic curvatures

        Returns:
            Tensor of shape B x d representing the element-wise Mobius addition of x and y.
        """
        x2 = torch.sum(x * x, dim=-1, keepdim=True)
        y2 = torch.sum(y * y, dim=-1, keepdim=True)
        xy = torch.sum(x * y, dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        return num / denom.clamp_min(self.MIN_NORM)

    def func(self, head, rel, tail, batch_type):
        # logging.info('shape head:{} tail:{}'.format(head.shape, tail.shape))
        head_0, head_1 = torch.chunk(head, 2, dim=2)
        rel_0, rel_1, rel_2 = torch.chunk(rel, 3, dim=2)
        tail_0, tail_1 = torch.chunk(tail, 2, dim=2)

        # if batch_type == BatchType.HEAD_BATCH:
        #     target_score = (head_0) + ((rel_0) - (tail_0))
        # else:
        #     target_score = ((head_0) + (rel_0)) - (tail_0)

       
        head_0 = head_0 / (self.embedding_range.item() / self.pi)
        rel_0 = rel_0 / \
            (self.embedding_range.item() / self.pi)
        tail_0 = tail_0 / (self.embedding_range.item() / self.pi)

        if batch_type == BatchType.HEAD_BATCH:
            phase_score = head_0 + (rel_0 - tail_0)
        else:
            phase_score = (head_0 + rel_0) - tail_0

        phase_score = torch.sum(
            torch.abs(torch.cos(phase_score/2)), dim=2) 

        # phase_score = torch.sum(
        #     torch.abs(torch.sin(phase_score / 2)), dim=2)
        # phase_score = 0

        # mod_score = head_1 + rel_1 - tail_1
        if batch_type == BatchType.HEAD_BATCH:
            tmp = self.project(self.mobius_add(self.expmap0(torch.sin(rel_0) * rel_1, self.curv), -self.expmap0(torch.sin(tail_0) * tail_1, self.curv)), self.curv)
            mod_score = self.mobius_add(tmp, self.expmap0(torch.sin(head_0) * head_1, self.curv), self.curv)
        else:
            tmp = self.project(self.mobius_add(self.expmap0(torch.sin(rel_0) * rel_1, self.curv), self.expmap0(torch.sin(head_0) * head_1, self.curv)), self.curv)
            mod_score = self.mobius_add(tmp, -self.expmap0(torch.sin(tail_0) * tail_1, self.curv), self.curv)

        mod_score = torch.norm(mod_score, p=2, dim=2)

        rel_1 = torch.abs(rel_1)
        rel_2 = torch.clamp(rel_2, max=1)
        indicator = (rel_2 < -rel_1)
        rel_2[indicator] = -rel_1[indicator]

        r_score = head_1 * (rel_1 + rel_2) - tail_1 * (1 - rel_2)

        r_score = torch.norm(r_score, dim=2) 

        # logging.info('shape :{}'.format(phase_score.shape))
        
        return self.gamma.item() - (self.p_weight * phase_score + self.m_weight * mod_score + self.m_weight * r_score)
        # return self.gamma.item() - (self.poin_dis(head_0, tail_0) - torch.norm(torch.matmul(rel, self.project_embedding), p=2, dim=2))

