import torch
_is_hit_cache = {}


def create_hit_matrix(scores, ground_truth, top_k):
    global _is_hit_cache
    cache_id = (id(scores), id(ground_truth))
    if top_k in _is_hit_cache and _is_hit_cache[top_k]['id'] == cache_id:
        return _is_hit_cache[top_k]['is_hit']
    else:
        device = scores.device
        _, col_indice = torch.topk(scores, top_k)
        row_indice = torch.zeros_like(col_indice) + torch.arange(
            scores.shape[0], device=device, dtype=torch.long).view(-1, 1)
        is_hit = ground_truth[row_indice.view(-1), col_indice.view(-1)].view(-1, top_k)
        _is_hit_cache[top_k] = {'id': cache_id, 'is_hit': is_hit}
        return is_hit


class Metrics:
    def __init__(self):
        self._cnt = 0
        self._metric = 0
        self._sum = 0
        self.start()

    def __call__(self, scores, ground_truth):
        raise NotImplementedError

    @property
    def metric(self):
        return self._metric

    def title(self):
        raise NotImplementedError

    def start(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._cnt = 0
        self._metric = 0
        self._sum = 0

    def stop(self):
        global _is_hit_cache
        _is_hit_cache = {}
        self._metric = self._sum/self._cnt


class Recall(Metrics):
    def __init__(self, top_k, eps=1e-8):
        super().__init__()
        self.top_k = top_k
        self.eps = eps

    def __call__(self, scores, ground_truth):

        is_hit = create_hit_matrix(scores, ground_truth, self.top_k)

        is_hit = is_hit.sum(dim=1)
        num_pos = ground_truth.sum(dim=1)

        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += (is_hit / (num_pos + self.eps)).sum().item()

    def title(self):
        return "Recall@{}".format(self.top_k)


class NDCG(Metrics):
    def __init__(self, top_k):
        super().__init__()
        self.top_k = top_k
        self.IDCGs_for_top_k = torch.empty(1 + self.top_k, dtype=torch.float)
        self.IDCGs_for_top_k[0] = 1
        for i in range(1, self.top_k + 1):
            self.IDCGs_for_top_k[i] = self.IDCG(i)

    def __call__(self, scores, ground_truth):
        device = scores.device
        is_hit = create_hit_matrix(scores, ground_truth, self.top_k)
        num_pos = ground_truth.sum(dim=1).clamp(0, self.top_k).to(torch.long)
        dcg = self.DCG(is_hit, device)
        idcg = self.IDCGs_for_top_k[num_pos]
        ndcg = dcg/idcg.to(device)
        self._cnt += scores.shape[0] - (num_pos == 0).sum().item()
        self._sum += ndcg.sum().item()

    def title(self):
        return "NDCG@{}".format(self.top_k)

    def DCG(self, hit, device=torch.device('cpu')):
        hit = hit.float()/torch.log2(torch.arange(2, self.top_k+2, device=device, dtype=torch.float32))
        return hit.sum(-1)

    def IDCG(self, num_pos):
        hit = torch.zeros(self.top_k, dtype=torch.float)
        hit[:num_pos] = 1
        return self.DCG(hit)
