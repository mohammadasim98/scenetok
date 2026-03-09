import torch

def generate_random_context_mask(shape, device, sharpness=1.0):

    b, v = shape

    # Exponentially decaying probabilities (high for later indices, low for earlier)
    probs = torch.arange(v - 1, -1, -1, dtype=torch.float32, device=device)  # [v-1, ..., 0]
    probs = torch.exp(probs * sharpness)  # sharper decay with larger `sharpness`
    probs /= probs.sum()  # normalize

    # Sample cutoff indices
    n = torch.multinomial(probs, num_samples=b, replacement=True)

    # Create mask
    range_row = torch.arange(v, device=device).unsqueeze(0)
    n_col = n.unsqueeze(1)
    mask = (range_row >= n_col)
    return mask

def generate_random_context_mask_tail_decay(shape, device, tail_decay=1.0):
    b, v = shape

    # Step 1: Create probability vector
    probs = torch.zeros(v, dtype=torch.float32, device=device)

    # Assign 10% probability to n = 0 (no zeros in mask)
    probs[0] = 0.10

    # Step 2: For n ≥ 1, use exponential decay
    tail_indices = torch.arange(1, v, device=device)
    tail_weights = torch.exp(-tail_indices * tail_decay)  # exponential decay
    tail_weights /= tail_weights.sum()  # normalize tail to sum to 1
    tail_weights *= 0.90  # scale tail to sum to 90%

    # Assign to rest of probs
    probs[1:] = tail_weights

    # Step 3: Sample cutoff index `n` for each row
    n = torch.multinomial(probs, num_samples=b, replacement=True)

    # Step 4: Construct mask
    range_row = torch.arange(v, device=device).unsqueeze(0)  # shape (1, v)
    n_col = n.unsqueeze(1)  # shape (b, 1)
    mask = (range_row >= n_col)  # shape (b, v)

    return mask


def blockwise_random_mask(shape, min_false: int) -> torch.BoolTensor:
    """
    Creates a mask of shape (b, v) where each row is a sequence of
    leading Falses followed by trailing Trues.
    Each row has at least `min_false` Falses (i.e., max `v - min_false` Trues).

    Args:
        b (int): Batch size
        v (int): Number of elements per row
        min_false (int): Minimum number of False values per row

    Returns:
        torch.BoolTensor: Boolean mask of shape (b, v)
    """
    b, v = shape
    assert 0 <= min_false <= v, "min_false must be in [0, v - 1]"
    max_n_true = v - min_false

    mask = torch.zeros((b, v), dtype=torch.bool)
    for i in range(b):
        n_true = torch.randint(0, max_n_true + 1, (1,)).item()  # at least 1 True, at most v - min_false
        mask[i, -n_true:] = True if n_true > 0 else False  # avoid slicing with -0
    return mask

def generate_biased_boolean_mask(shape, min_false: int, true_prob=0.4):
    """
    Generate a boolean mask of shape (B, V) with:
      - at least `n` Falses per row
      - remaining values biased to be more often False than True

    Args:
        B (int): Number of rows (batch size)
        V (int): Number of elements per row
        n (int): Minimum number of False values per row
        true_prob (float): Probability of assigning True to each remaining position (0 < true_prob < 0.5 recommended)

    Returns:
        torch.Tensor: Boolean tensor of shape (B, V)
    """
    B, V = shape
    n = min_false  # Remaining positions after ensuring at least `n` Falses
    if n > V:
        raise ValueError("`n` must be <= `V`")
    if not (0.0 <= true_prob <= 1.0):
        raise ValueError("`true_prob` must be between 0 and 1")

    mask = torch.ones(B, V, dtype=torch.bool)

    for i in range(B):
        # Ensure at least n False values
        false_indices = torch.randperm(V)[:n]
        mask[i, false_indices] = False

        # Fill remaining indices with biased values
        remaining_indices = torch.tensor([j for j in range(V) if j not in false_indices])
        remaining_mask = torch.rand(len(remaining_indices)) < true_prob  # True with probability `true_prob`
        mask[i, remaining_indices] = remaining_mask

    return mask


def random_mask_biased(B: int, N: int, M: float, device=None):
    """
    Vectorized, safe mask generator.

    - Per-row mask ratio: m = max(0, U(-0.1, M))
    - num_false = floor(m * N)
    - For each row, mask exactly num_false items (set to False).

    Args:
        B (int): batch size
        N (int): number of tokens / patches per row
        M (float): maximum mask ratio (0 <= M <= 1)
        device: torch device or string (e.g. "cuda" or "cpu")

    Returns:
        mask (torch.BoolTensor): shape (B, N). True = keep / visible, False = masked.
        ratios (torch.Tensor): sampled m per row, shape (B,)
        num_false (torch.LongTensor): number of masked tokens per row, shape (B,)
    """
    device = torch.device(device) if device is not None else torch.device("cpu")

    # sample biased ratios m = max(0, U(-0.1, M))
    ratios = torch.rand(B, device=device) * (M + 0.1) - 0.1
    ratios = ratios.clamp(min=0.0, max=1.0)

    # compute number to mask per row (use floor to be explicit)
    num_false = (ratios * N).floor().long()  # in [0, N]

    # random scores per element
    rand_vals = torch.rand(B, N, device=device)

    # get rank of each element within its row: ranks in {0, ..., N-1}
    # two argsorts: first gives ordering indices, second gives ranks
    order = rand_vals.argsort(dim=1)
    ranks = order.argsort(dim=1)

    # keep elements whose rank >= num_false (i.e., the top N-num_false elements)
    mask = ranks >= num_false.unsqueeze(1)

    return mask, ratios, num_false