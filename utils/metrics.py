import torch
import numpy as np
import math

def compute_gap(feat_modality1: torch.Tensor, feat_modality2: torch.Tensor) -> float:
    """
    Compute the Euclidean distance between the centroids of two modalities.

    Args:
        feat_modality1 (torch.Tensor): Feature matrix of modality 1 with shape [N, D].
        feat_modality2 (torch.Tensor): Feature matrix of modality 2 with shape [N, D].

    Returns:
        float: Euclidean distance between centroids.
    """
    # Ensure features are normalized if required
    modality1_centroid = torch.mean(feat_modality1, dim=0)
    modality2_centroid = torch.mean(feat_modality2, dim=0)

    gap = modality1_centroid - modality2_centroid
    norm_gap = torch.norm(gap).item()

    return norm_gap

def compute_mean_angular_value_of_a_modality(feat_modality: torch.Tensor) -> float:
    """
    Compute the mean angular value (mean cosine similarity) of a modality.

    Args:
        feat_modality (torch.Tensor): Feature matrix with shape [N, D].

    Returns:
        float: Mean angular value.
    """
    # Compute cosine similarity matrix
    cos_sim = feat_modality @ feat_modality.T

    # Exclude diagonal elements by creating a mask
    mask = ~torch.eye(cos_sim.size(0), dtype=torch.bool, device=cos_sim.device)
    cos_sim_no_diag = cos_sim[mask]

    mean_cos_sim = cos_sim_no_diag.mean().item()

    return mean_cos_sim

def uniformity(features_modality1: torch.Tensor, features_modality2: torch.Tensor) -> float:
    x = torch.cat([features_modality1, features_modality2], dim=0)
    N = x.size(0)
    dim = x.size(1)

    x_center = torch.mean(x, dim=0, keepdim=True)
    covariance = torch.mm((x - x_center).t(), x - x_center) / N

    mean =  x.mean(0)
    np_mean = mean.data.cpu().numpy()
    np_covariance = covariance.data.cpu().numpy()
   
    ##calculation of part1
    part1 = np.sum(np.multiply(np_mean, np_mean))

    ##calculation of part2
    eps = 1e-8 
    S, Q = np.linalg.eig(np_covariance)
    S = S + eps

    mS = np.sqrt(np.diag(S.clip(min=0)))

    covariance_2 = np.dot(np.dot(Q, mS), Q.T)

    part2 = np.trace(np_covariance - 2.0/np.sqrt(dim) * covariance_2)
    wasserstein_distance = math.sqrt(part1 + 1 + part2)
    return -wasserstein_distance 


def mean_distance_of_true_pairs(features_modality1: torch.Tensor, features_modality2: torch.Tensor, cosine = True) -> float:
    """
    Compute the mean cosine similarity of true pairs between two modalities.

    Args:
        features_modality1 (torch.Tensor): Normalized feature matrix of modality 1 with shape [N, D].
        features_modality2 (torch.Tensor): Normalized feature matrix of modality 2 with shape [N, D].

    Returns:
        float: Mean cosine similarity of true pairs.
    """
    # Compute cosine similarity matrix
    if cosine:
        cosine_sim = torch.matmul(features_modality1, features_modality2.T)

        # Extract diagonal elements (true pairs)
        cosine_sim_diag = torch.diag(cosine_sim)

        # Compute mean cosine similarity of true pairs
        cosine_tv_mean = torch.mean(cosine_sim_diag).item()

        return cosine_tv_mean

    else:
        # Compute Euclidean distance matrix
        euclidean_dist = torch.cdist(features_modality1, features_modality2)

        # Extract diagonal elements (true pairs)
        euclidean_dist_diag = torch.diag(euclidean_dist)

        # Compute mean Euclidean distance of true pairs
        euclidean_tv_mean = torch.mean(euclidean_dist_diag).item()

        return euclidean_tv_mean



def compute_rmg(image_features, text_features) -> float:
    """
    
    Args:
        image_features (torch.Tensor): Normalized image feature matrix with shape [N, D].
        text_features (torch.Tensor): Normalized text feature matrix with shape [N, D].
        Returns:

    Returns:
        float: The computed RMG metric.
    """
    image_features_original = image_features.clone()
    #image_features = torch.stack([image_features[l] for l in text_to_image_map], dim=0)
    text_feature_per_image = text_features
    
    image_features_matching = torch.sum(image_features*text_feature_per_image, dim=1).mean()
    image_features_matching = 1-(image_features_matching+1)/2 # [0, 1] & flip
    image_features_matching = torch.where(image_features_matching > 0, image_features_matching, torch.ones_like(image_features_matching)*1e-3) # [1e-3, 1]

    
    i_x_i = image_features_original @ image_features_original.T

    i_x_i.fill_diagonal_(0)
    mean_img_similarity = i_x_i.sum() / (math.prod(i_x_i.shape)-i_x_i.shape[0]) # [-1, 1]
    mean_img_similarity = 1-(mean_img_similarity+1)/2 # [0,1] & flip

    t_x_t = text_features @ text_features.T
    t_x_t.fill_diagonal_(0)
    mean_txt_similarity = t_x_t.sum() / (math.prod(t_x_t.shape)-t_x_t.shape[0]) # [-1, 1]
    mean_txt_similarity = 1-(mean_txt_similarity+1)/2 # [0,1] & flip

    normalizer = image_features_matching.mean() + (mean_img_similarity.mean() + mean_txt_similarity.mean()) / 2
    dist = image_features_matching.mean().item() / normalizer.item()

    return dist