import torch
import time

# ========== BASE FUNCTIONS ==========

def volume_computation_takashi_3_optimized(language, video, audio):
    lv = language @ video.T
    la = language @ audio.T
    vv = (video * video).sum(-1)[None, :]
    aa = (audio * audio).sum(-1)[None, :]
    va = (video * audio).sum(-1)[None, :]
    ll = (language * language).sum(-1)[:, None]
    v1v1 = vv + ll - 2 * lv
    v2v2 = aa + ll - 2 * la
    v1v2 = va + ll - lv - la
    det = v1v1 * v2v2 - v1v2 ** 2
    res = torch.sqrt(torch.clamp(det, min=0.0)) / 6.0
    return res


def compute_attention_scores_looped(query_layer, key_1_layer, key_2_layer):
    attention_scores = []
    for i in range(query_layer.shape[0]):  # B
        attention_for_each_head = []
        for h in range(query_layer.shape[1]):  # H
            gram_scores = volume_computation_takashi_3_optimized(
                query_layer[i][h], key_1_layer[i][h], key_2_layer[i][h]
            )
            attention_for_each_head.append(-gram_scores)
        attention_for_each_head = torch.stack(attention_for_each_head, dim=0)
        attention_scores.append(attention_for_each_head)
    return torch.stack(attention_scores, dim=0)


import torch

def volume_computation3_batched_optimized(language, video, audio):
    """
    Optimized fully batched computation of 3x3 Gram volumes for all pairs
    between language_i and (video_j, audio_j) using efficient matmul and elementwise ops.

    Inputs:
        language: [B,H,N_lang,D]
        video:    [B,H,N_vid,D]
        audio:    [B,H,N_vid,D]
    Output:
        volumes:  [B,H,N_lang,N_vid]
    """
    B, H, N_lang, D = language.shape
    N_vid = video.shape[2]

    # Compute all necessary dot products using matmul or elementwise sum
    # language · language (ll) per sample
    ll = (language ** 2).sum(-1)[:, :, :, None]        # [B,H,N_lang,1]
    
    # video · video and audio · audio (vv, aa)
    vv = (video ** 2).sum(-1)[:, :, None, :]           # [B,H,1,N_vid]
    aa = (audio ** 2).sum(-1)[:, :, None, :]           # [B,H,1,N_vid]
    
    # pairwise dot products
    lv = torch.matmul(language, video.transpose(-1, -2))  # [B,H,N_lang,N_vid]
    la = torch.matmul(language, audio.transpose(-1, -2))  # [B,H,N_lang,N_vid]
    va = torch.matmul(video, audio.transpose(-1, -2))     # [B,H,N_lang,N_vid] ???

    # Wait — va is video_i · audio_j. We want **same index j** for all pairs:
    # Instead, we compute for broadcasted pair:
    # va_ij = video_j · audio_j repeated for all N_lang? Actually we need va[i,j] = v_j·a_j
    # Let's compute elementwise for video/audio only
    va = (video * audio).sum(-1)[:, :, None, :]          # [B,H,1,N_vid]

    # Compute the 2x2 Gram determinant for differences
    v1v1 = vv + ll - 2*lv          # (v-l)·(v-l)
    v2v2 = aa + ll - 2*la          # (a-l)·(a-l)
    v1v2 = va + ll - lv - la       # (v-l)·(a-l)
    
    # Determinant of 2x2 Gram matrix
    det = v1v1 * v2v2 - v1v2**2

    # Volume = sqrt(det) / 6 (for tetrahedron)
    res = torch.sqrt(torch.clamp(det, min=0.0)) / 6.0  # [B,H,N_lang,N_vid]

    return res


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


# ========== TESTING + BENCHMARKING ==========

def benchmark(func, *args, device='cpu', repeat=10):
    # Warmup
    for _ in range(3):
        _ = func(*args)

    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        for _ in range(repeat):
            _ = func(*args)
        ender.record()
        torch.cuda.synchronize()
        elapsed_ms = starter.elapsed_time(ender) / repeat
        return elapsed_ms
    else:
        start = time.perf_counter()
        for _ in range(repeat):
            _ = func(*args)
        end = time.perf_counter()
        return (end - start) * 1000 / repeat  # ms


if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, N, D = 8, 12, 64, 768  # medium-size test

    query = torch.randn(B, H, N, D)
    key1  = torch.randn(B, H, N, D)
    key2  = torch.randn(B, H, N, D)

    # CPU correctness check
    scores_loop = compute_attention_scores_looped(query, key1, key2)
    scores_parallel = compute_attention_scores_parallel(query, key1, key2)

    print(f"scores_loop shape: {scores_loop}")
    print(f"scores_parallel shape: {scores_parallel}")

    print("✅ Max abs diff:", (scores_loop - scores_parallel).abs().max().item())
    assert torch.allclose(scores_loop, scores_parallel, atol=1e-6)

    # CPU timing
    t_loop_cpu = benchmark(compute_attention_scores_looped, query, key1, key2, device='cpu')
    t_parallel_cpu = benchmark(compute_attention_scores_parallel, query, key1, key2, device='cpu')

    print(f"\n⏱ CPU Inference time per run:")
    print(f"   Looped:     {t_loop_cpu:.3f} ms")
    print(f"   Parallel:   {t_parallel_cpu:.3f} ms")
    print(f"   Speedup:    {t_loop_cpu/t_parallel_cpu:.2f}×")

    # GPU timing (if available)
    if torch.cuda.is_available():
        query, key1, key2 = query.cuda(), key1.cuda(), key2.cuda()
        t_loop_gpu = benchmark(compute_attention_scores_looped, query, key1, key2, device='cuda')
        t_parallel_gpu = benchmark(compute_attention_scores_parallel, query, key1, key2, device='cuda')
        print(f"\n⚙️  GPU Inference time per run:")
        print(f"   Looped:     {t_loop_gpu:.3f} ms")
        print(f"   Parallel:   {t_parallel_gpu:.3f} ms")
        print(f"   Speedup:    {t_loop_gpu/t_parallel_gpu:.2f}×")
