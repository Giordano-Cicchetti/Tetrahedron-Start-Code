import torch
import numpy as np

# * * * *  * * * *  * * * *   *       *   
# *        *     *  *     *   * *   * *   
# *   * *  * * *    * * * *   *   *   *   
# *     *  *     *  *     *   *       *   
# * * * *  *     *  *     *   *       *   

# THIS IS THE CORE PY CODE OF GRAM FRAMEWORK
import torch.nn.functional as F
import torch


def simple_volume_computation(language, video, audio):
    A = torch.stack([language, video, audio])
    G = A @ A.T
    gramian = torch.linalg.det(G)
    return torch.sqrt(gramian)

def simple_gram_matrix_computation(language, video, audio):
    A = torch.stack([language, video, audio])
    G = A @ A.T
    return G

def simple_tethraedron_matrix_computation(language, video, audio):

    v1 = video - language
    v2 = audio - language

    a = [v1, v2]
    a = torch.stack(a, dim=0)  # shape: [batch_size, 3, feature_dim]
    G = a @ a.T  # shape: [batch_size, 3, 3]
    return G


def volume_computation_takashi_3_optimized(language, video, audio):
    """
    Memory-efficient version of Takashi GPU function.
    Computes simplex volume without expanding to [B, N, D].
    """

    # language: [B, D]
    # video, audio: [N, D]

    # Compute pairwise dot products using matmul (no broadcasting!)
    lv = language @ video.T      # [B, N]
    la = language @ audio.T      # [B, N]
    vv = (video * video).sum(-1)[None, :]  # [1, N]
    aa = (audio * audio).sum(-1)[None, :]  # [1, N]
    va = (video * audio).sum(-1)[None, :]  # [1, N]
    ll = (language * language).sum(-1)[:, None]  # [B, 1]

    # Compute dot products for the *differences*
    v1v1 = vv + ll - 2 * lv      # (v - l)·(v - l)
    v2v2 = aa + ll - 2 * la      # (a - l)·(a - l)
    v1v2 = va + ll - lv - la     # (v - l)·(a - l)

    # Build 2x2 Gram matrix per pair
    G11 = v1v1
    G12 = v1v2
    G22 = v2v2

    # Determinant of 2x2 matrix = G11*G22 - G12^2
    det = G11 * G22 - G12 ** 2

    # Volume = sqrt(det) / 6 (for tetrahedron)
    res = torch.sqrt(torch.clamp(det, min=0.0)) / 6.0

    return res  # [B, N]

def volume_computation_takashi_gpu_3(language, video, audio):
    """
    Compute the 4D simplex (tetrahedron) volume spanned by
    language, video, audio, and subtitles embeddings.

    Args:
        language:  (B, D)
        video:     (N, D)
        audio:     (N, D)
        subtitles: (N, D)
    Returns:
        volumes: (B, N)
    """

    # Compute differences: shape -> (B, N, D)
    v1 = video.unsqueeze(0) - language.unsqueeze(1)
    v2 = audio.unsqueeze(0) - language.unsqueeze(1)

    # Stack: (B, N, 2, D)
    a = torch.stack([v1, v2], dim=2)

    # Compute Gram matrix G = A A^T for each (B, N)
    # Result shape: (B, N, 2, 2)
    G = torch.matmul(a, a.transpose(-1, -2))

    # Determinant of each Gram matrix
    gram_det = torch.det(G.float())

    # Volume (normalized by 4! = 24)
    res = torch.sqrt(torch.abs(gram_det)) / 6

    return res  # shape (B, N)
def volume_computation_takashi_single3(language, video, audio):
    print("language shape:", language.shape)
    print("video shape:", video.shape)
    print("audio shape:", audio.shape)

    v1 = video - language
    v2 = audio - language


    a = [v1, v2]
    a = torch.stack(a, dim=0)  # shape: [batch_size, 3, feature_dim]
    print("a shape:", a.shape)
    G = a @ a.T  # shape: [batch_size, 3, 3]
    print("G shape:", G.shape)
    gram_det = torch.det(G.float())
    res = torch.sqrt(torch.abs(gram_det))
    res = res/6 # normalize by 4! = 24
    print("res shape:", res.shape)
    print("res:", res)
    return res

def volume_computation_takashi_single(language, video, audio, subtitles):

    v1 = video - language
    v2 = audio - language
    v3 = subtitles - language


    a = [v1, v2, v3]
    a = torch.stack(a, dim=0)  # shape: [batch_size, 3, feature_dim]
    G = a @ a.T  # shape: [batch_size, 3, 3]
    gram_det = torch.det(G.float())
    res = torch.sqrt(torch.abs(gram_det))
    res = res/24 # normalize by 4! = 24
    return res

feat_t = torch.randn(4, 768)  #(batch_size, feature_dim)
feat_v = torch.randn(10, 768)  #(num_frames, feature_dim)
feat_a = torch.randn(10, 768)  #(num_frames, feature_dim)
feat_s = torch.randn(10, 768)  #(num_frames, feature_dim)

feat_t = F.normalize(feat_t, p=2, dim=-1)
feat_v = F.normalize(feat_v, p=2, dim=-1)
feat_a = F.normalize(feat_a, p=2, dim=-1)
feat_s = F.normalize(feat_s, p=2, dim=-1)

volume = []

for i in range(feat_t.shape[0]):
    #print("feat_v_all sample shape:", feat_v_all[i].shape)
    volume_i = []
    
    for j in range(feat_v.shape[0]):


        volume_ij = volume_computation_takashi_single(feat_t[i], feat_v[j], feat_a[j], feat_s[j])  #(feat_t,feat_v,feat_a)
        volume_i.append(volume_ij)

        #min_volume = min(volume_single_frames)
    volume_tensor = torch.stack(volume_i).squeeze(-1)
    volume.append(volume_tensor)

volume = torch.stack(volume)
#print("volume shape:", volume.shape)  #(batch_size, num_frames)S
#print(volume)


def area_computation(language, video, audio):


    #print(f"norm language= {torch.sum(language ** 2, dim=1)}")
    
    language_expanded = language.unsqueeze(1)  # Shape: (n, 1, dim)

    # Compute the differences for all pairs (i-th language embedding with all j-th video/audio embeddings)
    u = language_expanded - video.unsqueeze(0)  # Shape: (n, n, dim)
    v = language_expanded - audio.unsqueeze(0)  # Shape: (n, n, dim)

    # Compute the norms for u and v
    u_norm = torch.sum(u ** 2, dim=2)  # Shape: (n, n)
    v_norm = torch.sum(v ** 2, dim=2)  # Shape: (n, n)

    # Compute the dot products for all pairs
    uv_dot = torch.sum(u * v, dim=2)  # Shape: (n, n)

    # Calculate the area for all pairs. I remove sqrt calculation
    area = ((u_norm * v_norm) - (uv_dot ** 2))/2#torch.sqrt((u_norm * v_norm) - (uv_dot ** 2)) / 2  # Shape: (n, n)
    
    return area

import torch
import torch.nn.functional as F

def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def volume_computation_takashi_gpu(language, video, audio, subtitles):
    """
    Compute the 4D simplex (tetrahedron) volume spanned by
    language, video, audio, and subtitles embeddings.

    Args:
        language:  (B, D)
        video:     (N, D)
        audio:     (N, D)
        subtitles: (N, D)
    Returns:
        volumes: (B, N)
    """

    # Compute differences: shape -> (B, N, D)
    v1 = video.unsqueeze(0) - language.unsqueeze(1)
    v2 = audio.unsqueeze(0) - language.unsqueeze(1)
    v3 = subtitles.unsqueeze(0) - language.unsqueeze(1)

    # Stack: (B, N, 3, D)
    a = torch.stack([v1, v2, v3], dim=2)

    # Compute Gram matrix G = A A^T for each (B, N)
    # Result shape: (B, N, 3, 3)
    G = torch.matmul(a, a.transpose(-1, -2))

    # Determinant of each Gram matrix
    gram_det = torch.det(G.float())

    # Volume (normalized by 4! = 24)
    res = torch.sqrt(torch.abs(gram_det)) / 24.0

    return res  # shape (B, N)


volume = volume_computation_takashi_gpu(feat_t, feat_v, feat_a, feat_s)
#print("volume shape (GPU):", volume.shape)  #(batch_size, num_frames)
#print(volume)


volume = volume_computation_takashi_3_optimized(feat_t, feat_v, feat_a)
#print("volume shape (GPU 3 optimized):", volume.shape)  #(batch_size, num_frames)
#print(volume)

volume = volume_computation_takashi_gpu_3(feat_t, feat_v, feat_a)
#print("volume shape (GPU 3):", volume.shape)  #(batch_size, num_frames)
#print(volume)





def volume_computation3(language, video, audio):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 3x3
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 3, 3])
    G = torch.stack([
        torch.stack([ll, lv, la], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa], dim=-1)  # Third row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def volume_computation4(language, video, audio, subtitles):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la, ls], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss], dim=-1)   # Fourth row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def volume_computation4_takashi(language, video, audio, subtitles):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 4x4
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 4, 4])
    G = torch.stack([
        torch.stack([ll, lv, la, ls], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss], dim=-1)   # Fourth row of the Gram matrix
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res

def volume_computation5(language, video, audio, subtitles, depth):

    """
    Computes the volume for each pair of samples between language (shape [batch_size1, feature_dim])
    and video, audio, subtitles (shape [batch_size2, feature_dim]) using the determinant of a 5x5
    Gram matrix.
    
    Parameters:
    - language (torch.Tensor): Tensor of shape (batch_size1, feature_dim) representing language features.
    - video (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing video features.
    - audio (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing audio features.
    - subtitles (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing subtitle features.
    - depth (torch.Tensor): Tensor of shape (batch_size2, feature_dim) representing depth features.    
    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """

    batch_size1 = language.shape[0]  # For language
    batch_size2 = video.shape[0]     # For video, audio, subtitles

    # Compute pairwise dot products for language with itself (shape: [batch_size1, 1])
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with video, audio, and subtitles (shape: [batch_size1, batch_size2])
    lv = language@video.T
    la = language@audio.T
    ls = language@subtitles.T
    ld = language@depth.T

    # Compute pairwise dot products for video, audio, and subtitles with themselves and with each other
    vv = torch.einsum('bi,bi->b', video, video).unsqueeze(0).expand(batch_size1, -1)
    va = torch.einsum('bi,bi->b', video, audio).unsqueeze(0).expand(batch_size1, -1)
    aa = torch.einsum('bi,bi->b', audio, audio).unsqueeze(0).expand(batch_size1, -1)
    
    
    ss = torch.einsum('bi,bi->b', subtitles, subtitles).unsqueeze(0).expand(batch_size1, -1)
    vs = torch.einsum('bi,bi->b', video, subtitles).unsqueeze(0).expand(batch_size1, -1)
    sa = torch.einsum('bi,bi->b', audio, subtitles).unsqueeze(0).expand(batch_size1, -1)

    dd = torch.einsum('bi,bi->b', depth, depth).unsqueeze(0).expand(batch_size1, -1)
    dv = torch.einsum('bi,bi->b', depth, video).unsqueeze(0).expand(batch_size1, -1)
    da = torch.einsum('bi,bi->b', depth, audio).unsqueeze(0).expand(batch_size1, -1) 
    ds = torch.einsum('bi,bi->b', depth, subtitles).unsqueeze(0).expand(batch_size1, -1)


    # Stack the results to form the Gram matrix for each pair (shape: [batch_size1, batch_size2, 5, 5])
    G = torch.stack([
        torch.stack([ll, lv, la, ls, ld], dim=-1),  # First row of the Gram matrix
        torch.stack([lv, vv, va, vs, dv], dim=-1),  # Second row of the Gram matrix
        torch.stack([la, va, aa, sa, da], dim=-1),  # Third row of the Gram matrix
        torch.stack([ls, vs, sa, ss, ds], dim=-1),   # Fourth row of the Gram matrix
        torch.stack([ld, dv, da, ds, dd], dim=-1)
    ], dim=-2)

    # Compute the determinant for each Gram matrix (shape: [batch_size1, batch_size2])
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    #print(res.shape)
    return res


def volume_computation(language, *inputs):
    """
    General function to compute volume for contrastive learning loss functions.
    Compute the volume metric for each vector in language batch and all the other modalities listed in *inputs.

    Args:
    - language (torch.Tensor): Tensor of shape (batch_size1, dim)
    - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    batch_size1 = language.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Compute pairwise dot products for language with itself
    ll = torch.einsum('bi,bi->b', language, language).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with each input
    l_inputs = [language @ input.T for input in inputs]

    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)

    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([ll] + l_inputs, dim=-1),
        *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
    ], dim=-2)

    # Compute the determinant for each Gram matrix
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    return res




def multimodal_volume(x):
    """
    x: Tensor of shape (B, N, D)
       B = batch size
       N = number of inputs/modalities
       D = latent dimension

    Returns:
        volumes: shape (B,)
    """
    # x already contains all modalities → treat it as A
    # A = (B, N, D)
    A = x

    # Compute Gram matrix G = A A^T → (B, N, N)
    G = A @ A.transpose(-1, -2)

    # Determinant for each batch element (B,)
    detG = torch.linalg.det(G.float())

    # Volume = sqrt(det(G))
    volume = torch.sqrt(detG.clamp(min=0))

    return volume


B, N, D = 8, 8, 128   # 5 modalities
x = torch.randn(B, N, D)
x = F.normalize(x, dim=-1)

volumes = multimodal_volume(x)
print(volumes)         # tensor of shape (B,)
print(volumes.shape)   # torch.Size([8])