import torch
import time

# ========== ORIGINAL PER-SAMPLE VERSION ==========
def volume_computation3(language, video, audio):
    """
    Computes the 3x3 Gram matrix volume for each pair (language_i, video_j, audio_j)
    language: [B1, D]
    video, audio: [B2, D]
    Returns: [B1, B2]
    """
    batch_size1 = language.shape[0]
    batch_size2 = video.shape[0]

    ll = (language * language).sum(-1).unsqueeze(1).expand(-1, batch_size2)
    lv = language @ video.T
    la = language @ audio.T
    vv = (video * video).sum(-1).unsqueeze(0).expand(batch_size1, -1)
    va = (video * audio).sum(-1).unsqueeze(0).expand(batch_size1, -1)
    aa = (audio * audio).sum(-1).unsqueeze(0).expand(batch_size1, -1)

    G = torch.stack([
        torch.stack([ll, lv, la], dim=-1),
        torch.stack([lv, vv, va], dim=-1),
        torch.stack([la, va, aa], dim=-1)
    ], dim=-2)

    gram_det = torch.det(G.float())
    return torch.sqrt(torch.abs(gram_det))


# ========== LOOPED BASELINE ==========
def compute_volume3_looped(query_layer, key_1_layer, key_2_layer):
    """
    query_layer, key_1_layer, key_2_layer: [B, H, N, D]
    Returns: [B, H, N]
    """
    results = []
    for b in range(query_layer.shape[0]):
        heads = []
        for h in range(query_layer.shape[1]):
            base = volume_computation3(query_layer[b][h], key_1_layer[b][h], key_2_layer[b][h])
            diag = base  # only corresponding (i,i) samples
            heads.append(diag)
        results.append(torch.stack(heads, dim=0))
    return torch.stack(results, dim=0)  # [B,H,N]


# ========== FIXED PARALLELIZED VERSION ==========
import torch

import torch

def volume_computation3_batched(language, video, audio):
    """
    Compute 3x3 Gram volumes for all pairs of language_i with (video_j, audio_j).
    language: [B,H,N_lang,D]
    video, audio: [B,H,N_vid,D]
    Returns: [B,H,N_lang,N_vid]
    """
    B, H, N_lang, D = language.shape
    N_vid = video.shape[2]

    # Expand dimensions for broadcasting
    l_exp = language[:, :, :, None, :]   # [B,H,N_lang,1,D]
    v_exp = video[:, :, None, :, :]      # [B,H,1,N_vid,D]
    a_exp = audio[:, :, None, :, :]      # [B,H,1,N_vid,D]

    # Compute pairwise dot products
    ll = (l_exp * l_exp).sum(-1)         # [B,H,N_lang,1]
    lv = (l_exp * v_exp).sum(-1)         # [B,H,N_lang,N_vid]
    la = (l_exp * a_exp).sum(-1)         # [B,H,N_lang,N_vid]
    vv = (v_exp * v_exp).sum(-1)         # [B,H,1,N_vid]
    va = (v_exp * a_exp).sum(-1)         # [B,H,1,N_vid]
    aa = (a_exp * a_exp).sum(-1)         # [B,H,1,N_vid]

    # Broadcast ll, vv, va, aa to shape [B,H,N_lang,N_vid]
    ll = ll.expand(-1, -1, -1, N_vid)
    vv = vv.expand(-1, -1, N_lang, -1)
    va = va.expand(-1, -1, N_lang, -1)
    aa = aa.expand(-1, -1, N_lang, -1)

    # Build 3x3 Gram matrix per pair
    G_row1 = torch.stack([ll, lv, la], dim=-1)
    G_row2 = torch.stack([lv, vv, va], dim=-1)
    G_row3 = torch.stack([la, va, aa], dim=-1)
    G = torch.stack([G_row1, G_row2, G_row3], dim=-2)  # [B,H,N_lang,N_vid,3,3]

    # Compute determinant per pair
    gram_det = torch.det(G.float())      # [B,H,N_lang,N_vid]

    return torch.sqrt(torch.abs(gram_det))




def compute_volume3_parallel(query_layer, key_1_layer, key_2_layer):
    return volume_computation3_batched(query_layer, key_1_layer, key_2_layer)

def volume_computation_takashi_3_optimized_batched(language, video, audio):
    """
    Fully batched and parallelized version.
    language, video, audio: [B, H, N, D]
    Returns: [B, H, N, N]
    """
    lv = torch.matmul(language, video.transpose(-1, -2))  # [B,H,N,N]
    la = torch.matmul(language, audio.transpose(-1, -2))  # [B,H,N,N]
    vv = (video * video).sum(-1)[:, :, None, :]  # [B,H,1,N]
    aa = (audio * audio).sum(-1)[:, :, None, :]  # [B,H,1,N]
    va = (video * audio).sum(-1)[:, :, None, :]  # [B,H,1,N]
    ll = (language * language).sum(-1)[:, :, :, None]  # [B,H,N,1]
    v1v1 = vv + ll - 2 * lv
    v2v2 = aa + ll - 2 * la
    v1v2 = va + ll - lv - la
    det = v1v1 * v2v2 - v1v2 ** 2
    res = torch.sqrt(torch.clamp(det, min=0.0)) / 6.0
    return res


def compute_attention_scores_parallel(query_layer, key_1_layer, key_2_layer):
    gram_scores = volume_computation_takashi_3_optimized_batched(
        query_layer, key_1_layer, key_2_layer
    )
    return -gram_scores
# ========== BENCHMARK HELPER ==========
def benchmark(func, *args, device='cpu', repeat=5):
    for _ in range(2):
        _ = func(*args)
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        for _ in range(repeat):
            _ = func(*args)
        ender.record()
        torch.cuda.synchronize()
        return starter.elapsed_time(ender) / repeat
    else:
        start = time.perf_counter()
        for _ in range(repeat):
            _ = func(*args)
        end = time.perf_counter()
        return (end - start) * 1000 / repeat


# ========== TEST ==========
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 2, 3, 8, 16

    query = torch.randn(B, H, N, D)
    key1  = torch.randn(B, H, N, D)
    key2  = torch.randn(B, H, N, D)

    

    scores_loop = compute_volume3_looped(query, key1, key2)
    scores_parallel = compute_volume3_parallel(query, key1, key2)
    scores_takashi = compute_attention_scores_parallel(query, key1, key2)

    print(f"scores_loop shape: {scores_loop.shape}")
    print(f"scores_parallel shape: {scores_parallel.shape}")
    print(f"scores_takashi shape: {scores_takashi.shape}")

    diff = (scores_loop - scores_parallel).abs().max().item()
    print(f"✅ Max abs diff: {diff:.8f}")
    assert torch.allclose(scores_loop, scores_parallel, atol=1e-6)

    # CPU timing
    t_loop_cpu = benchmark(compute_volume3_looped, query, key1, key2, device='cpu')
    t_parallel_cpu = benchmark(compute_volume3_parallel, query, key1, key2, device='cpu')

    print(f"\n⏱ CPU Inference time per run:")
    print(f"   Looped:     {t_loop_cpu:.3f} ms")
    print(f"   Parallel:   {t_parallel_cpu:.3f} ms")
    print(f"   Speedup:    {t_loop_cpu/t_parallel_cpu:.2f}×")

    if torch.cuda.is_available():
        query, key1, key2 = query.cuda(), key1.cuda(), key2.cuda()
        t_parallel_gpu = benchmark(compute_volume3_parallel, query, key1, key2, device='cuda')
        print(f"\n⚙️  GPU Inference time per run:")
        print(f"   Parallel (GPU): {t_parallel_gpu:.3f} ms")
