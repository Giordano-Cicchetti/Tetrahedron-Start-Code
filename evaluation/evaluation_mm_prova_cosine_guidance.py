import os
import json
import torch
import numpy as np
from time import time
import torch.distributed as dist
from tqdm import tqdm 
from torch.nn import functional as F
from evaluation_tools.caption_tools.pycocoevalcap.eval import COCOEvalCap
from evaluation_tools.caption_tools.pycocotools.coco import COCO
from evaluation_tools.vqa_tools.vqa import VQA
from evaluation_tools.vqa_tools.vqa_eval import VQAEval
from utils.logger import LOGGER
from utils.distributed import  all_gather_list, ddp_allgather
from utils.tool import NoOp
from easydict import EasyDict as edict
from utils.volume import volume_computation4,volume_computation3, volume_computation5
import wandb


def evaluate_mm(model, val_dataloaders, run_cfg, global_step):

    eval_log = {}
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"evaluate on {task} task")
        val_log = evaluate_single(model, loader, task.split('--')[0], run_cfg, global_step,task.split('--')[1])
        eval_log[task] = val_log
    model.train()
    return eval_log


@torch.no_grad()
def evaluate_single(model, val_loader, task, run_cfg, global_step,dset_name):
    LOGGER.info("start running {} validation...".format(task))

    tasks = task.split('_')

    output_ls = []

    for task in tasks:
        if task.startswith('ret'):
            ret_dict = evaluate_ret(model, task, val_loader, global_step)
            output_ls.append(ret_dict)
        elif task.startswith('cap'):
            cap_dict = evaluate_cap(model, task, val_loader, run_cfg, global_step, dset_name)
            output_ls.append(cap_dict)
        elif task.startswith('qa'):
            qa_dict = evaluate_qa(model, task, val_loader, run_cfg, global_step, dset_name)
            output_ls.append(qa_dict)

    output_dict = {k:v for dic in output_ls for k,v in dic.items() }
    return output_dict

@torch.no_grad()
def evaluate_qa(model, tasks, eval_loader, run_cfg, global_step, dset_name):
    val_log = {}
    output_dir = os.path.join(run_cfg.output_dir,f'predict_answers')
    os.makedirs(output_dir,exist_ok=True)
    subtasks = tasks.split('%')[1:]
    groundtruth_answers=[]
    store_dict = {}
    for task in subtasks:
        store_dict[f'generated_answers_{task}'] = []

    if dist.get_rank() == 0:
        pbar = tqdm(total=len(eval_loader))
    else:
        pbar = NoOp()

    for batch in eval_loader:
        batch = edict(batch)
        ids = batch.ids
        groundtruth_answers += [j for i in batch.raw_answers for j in i]
      
        evaluation_dict = model(batch, tasks, compute_loss=False)

        for task in subtasks:
            store_dict[f'generated_answers_{task}'] += evaluation_dict[f'generated_answers_{task}']

        pbar.update(1)
        
    pbar.close()

    groundtruth_answers = [i for j in all_gather_list(groundtruth_answers)  for i in j]
    total_num = len(groundtruth_answers)
    LOGGER.info('total {} questions has been tested'.format(total_num))

    for task in subtasks:
        store_dict[f'generated_answers_{task}'] = [i for j in all_gather_list(store_dict[f'generated_answers_{task}'])  for i in j]

        if dist.get_rank()==0:
            pred_file = os.path.join(output_dir,f'step{global_step}_pred_{dset_name}_{task}.json')
            json.dump(store_dict[f'generated_answers_{task}'],open(pred_file,'w'))

        accurate_num = sum([store_dict[f'generated_answers_{task}'][i] == groundtruth_answers[i] for i in range(total_num)])
        accuracy = accurate_num / total_num
        val_log[f'vqa_{task}'] = {'accuracy':round(accuracy*100,2)} 

    return val_log



@torch.no_grad()
def evaluate_cap(model, tasks, eval_loader, run_cfg, global_step, dset_name):
    val_log = {}
    captioner_mode = model.config.captioner_mode
    generate_nums =  model.config.generate_nums
    result_folder = os.path.join(run_cfg.output_dir, f'results_test_{dset_name}')
    os.makedirs(result_folder, exist_ok=True)
    subtasks = tasks.split('%')[1:]
    store_dict = {}
    if captioner_mode:
        for task in subtasks:
            store_dict[f'generated_captions_{task}'] = {}

    else:
        for task in subtasks:
            store_dict[f'generated_captions_{task}'] = []


    if dist.get_rank() == 0:
        pbar = tqdm()
    else:
        pbar = NoOp()

  
    gen_idx = 0
    for batch in eval_loader:
        batch = edict(batch)
        ids = batch.ids
        evaluation_dict = model(batch, tasks, compute_loss=False)

        for task in subtasks:
            sents = evaluation_dict[f'generated_captions_{task}']       
            if not captioner_mode:    
                for i in range(len(sents)):
                    store_dict[f'generated_captions_{task}'].append({'video_id':ids[i], 'caption': sents[i]})
                
            else:
                for i in range(len(ids)):
                    store_dict[f'generated_captions_{task}'][ids[i]] = sents[i*generate_nums : (i+1)*generate_nums]
                if  len(store_dict[f'generated_captions_{task}']) > 20000:
                    rank= dist.get_rank()
                    json.dump(store_dict[f'generated_captions_{task}'],open(os.path.join(result_folder, f'gencap_rank{rank}_idx{gen_idx}_{task}.json'), 'w'))
                    gen_idx+=1
                    store_dict[f'generated_captions_{task}'] = {}
            pbar.update(1)
    
    if captioner_mode:
        for task in subtasks:
            if len(store_dict[f'generated_captions_{task}']) > 0:
                rank= dist.get_rank()
                json.dump(store_dict[f'generated_captions_{task}'],open(os.path.join(result_folder, f'gencap_rank{rank}_idx{gen_idx}_{task}.json'), 'w'))
                gen_idx+=1
        return val_log


    annfile_path = eval_loader.dataset.annfile
    pbar.close()

    for task in subtasks:
        results = [i for j in all_gather_list(store_dict[f'generated_captions_{task}'])  for i in j]
        if dist.get_rank()==0:
            val_log[f'cap_{task}'] = compute_metric_cap(results, annfile_path) 
            json.dump(results,open(os.path.join(result_folder, f'step_{global_step}_{task}.json'), 'w'))
    
    return val_log



@torch.no_grad()
def evaluate_ret(model, tasks, val_loader, global_step):
    val_log = {}
    ids = []
    ids_txt = []
    input_ids = []
    attention_mask = []
 
    subtasks = tasks.split('%')[1:]
    store_dict = {}
    feat_t = []
    feat_a = []
    feat_v = []
    feat_s = []
    feat_d = []
    feat_v_all = []

    for task in subtasks:
        # store_dict[f'feat_cond_{task}'] = []
        store_dict[f'condition_feats_{task}'] = []    

    store_dict["condition_feats_v_not_reshaped"] = []
    num_batch = 0    

    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        num_batch += 1
        if num_batch == 5 :
            break
        batch = edict(batch)
        evaluation_dict= model(batch, tasks, compute_loss=False)
        # evaluation_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in evaluation_dict.items()}

        feat_t.append(evaluation_dict['feat_t'])
        feat_a.append(evaluation_dict['feat_a'])
        feat_v.append(evaluation_dict['feat_v'])
        if 'feat_s' in evaluation_dict.keys():
            feat_s.append(evaluation_dict['feat_s'])
        if 'feat_d' in evaluation_dict.keys():
            feat_d.append(evaluation_dict['feat_d'])

        feat_v_all.append(evaluation_dict['feat_v_all'])


        
      
        input_ids.append(evaluation_dict['input_ids'])
        attention_mask.append(evaluation_dict['attention_mask'])
        ids += batch.ids

        if 'ids_txt' in batch:
            if isinstance(batch['ids_txt'][0],list):
                ids_txt  +=  [j for i in batch.ids_txt for j in i]
            else:
                ids_txt  += batch.ids_txt
        else:    
            ids_txt  += batch.ids

  
        for task in subtasks:
            # store_dict[f'feat_cond_{task}'].append(evaluation_dict[f'feat_cond_{task}'])    
            store_dict[f'condition_feats_{task}'].append(evaluation_dict[f'condition_feats_{task}'])

        store_dict["condition_feats_v_not_reshaped"].append(evaluation_dict["condition_feats_v_not_reshaped"])

        
            
    ids = [j for i in all_gather_list(ids) for j in i]
    ids_txt = [j for i in all_gather_list(ids_txt) for j in i]
    input_ids = torch.cat([i for i in input_ids],dim=0)
    input_ids = ddp_allgather(input_ids)
    attention_mask = torch.cat([i for i in attention_mask],dim=0)
    attention_mask = ddp_allgather(attention_mask)
        
    feat_t = torch.cat(feat_t, dim = 0)
    feat_t = ddp_allgather(feat_t)

    feat_a = torch.cat(feat_a, dim = 0)
    feat_a = ddp_allgather(feat_a)

    feat_v = torch.cat(feat_v, dim = 0)
    feat_v = ddp_allgather(feat_v)

    feat_v_all = torch.cat(feat_v_all, dim=0)
    feat_v_all = ddp_allgather(feat_v_all)

    print("feat_v_all shape:", feat_v_all.shape)

    if len(feat_s)>0:
        feat_s = torch.cat(feat_s, dim = 0)
        feat_s = ddp_allgather(feat_s)

    if len(feat_d)>0:
        feat_d = torch.cat(feat_d, dim = 0)
        feat_d = ddp_allgather(feat_d)
    # torch.save(feat_t,f"./experiments/{global_step}text_features_msrvtt.pt")
    # torch.save(feat_v,f"./experiments/{global_step}video_features_msrvtt.pt")
    # torch.save(feat_a,f"./experiments/{global_step}audio_features_msrvtt.pt")
    #torch.save(feat_s,f"./experiments/{global_step}subtitles_features_msrvtt.pt")
    #if len(feat_s)>0:
    #    if len(feat_d)>0:
    #        area = volume_computation5(feat_t,feat_v,feat_a,feat_s,feat_d)
    #    else:
    #        area = volume_computation4(feat_t,feat_v,feat_a,feat_s) #(feat_t,feat_v,feat_a)
    #else:
    #    area = volume_computation3(feat_t,feat_v,feat_a)
    #bottom_k = 8 
    #volume = []
    #selected_frame_num = []
    #for i in range(feat_t.shape[0]):
    #    #print("feat_v_all sample shape:", feat_v_all[i].shape)
    #    volume_i = []
    #    selected_frame_num_i = []
    #    for j in range(feat_v_all.shape[0]):
#
    #        volume_single_frames = []
    #        for frame_feat in feat_v_all[j]:
    #            volume_frame = volume_computation4(feat_t[i].unsqueeze(0), frame_feat.unsqueeze(0), feat_a[j].unsqueeze(0), feat_s[j].unsqueeze(0))  #(feat_t,feat_v,feat_a)
    #            volume_single_frames.append(volume_frame)
#
    #        #min_volume = min(volume_single_frames)
    #        volume_tensor = torch.stack(volume_single_frames).squeeze(-1).squeeze(-1)
    #        #print(volume_tensor)
    #        #print(volume_tensor.shape)
    #        bottom_values, bottom_indices = torch.topk(volume_tensor, bottom_k, largest=False)
    #        #min_index = volume_single_frames.index(bottom_k)
#
    #        selected_frame_num_i.append(bottom_indices.tolist())
    #        #rint(bottom_values)
    #        #rint(bottom_values[0])
    #        volume_i.append(bottom_values[0].unsqueeze(0))
    #        #print(f"Processed volume for sample {i}-{j}: {volume_single_frames}")
#
    #    #prit(volume_i)
    #    volume.append(volume_i)
    #    selected_frame_num.append(selected_frame_num_i)
        
    #print(volume[0])
    #area = torch.cat([torch.cat(volume[i],dim=0).unsqueeze(0) for i in range(len(volume))],dim=0).squeeze(-1)
    #selected_frame_num = torch.tensor(selected_frame_num)
    volume_i = []
    for i in range(feat_v_all.shape[1]):
        volume = volume_computation4(feat_t, feat_v_all[:,i,:].squeeze(1), feat_a, feat_s)
        volume_i.append(volume)

    volume_cat = torch.stack(volume_i, dim=-1)
    
    volume, _ = torch.min(volume_cat, dim=2)


    bottom_k = 1
    _ , selected_frame_num = torch.topk(volume_cat, bottom_k, dim=2, largest=False)
    #selected_frame_num = selected_frame_num.squeeze(-1)



    print("Selected frame indices for minimum volume:", selected_frame_num)
    print("Selected frame shape:", selected_frame_num.shape)
    print("volume shape:", volume.shape)
    min_values_volume = torch.min(volume, 1).values
    mean_values_volume = torch.mean(min_values_volume)
    val_log[f"gramian_value"] = {"value": mean_values_volume.item()}
    
    
    log = compute_metric_ret_area(volume, ids, ids_txt, direction='forward')
    log = {k.replace('forward','volume_T2D'): v for k,v in log.items()}

    val_log[f'ret_area_forward'] = log

    log = compute_metric_ret_area(volume.T, ids, ids_txt, direction='forward')
    log = {k.replace('backward','volume_D2T'): v for k,v in log.items()}

    val_log[f'ret_area_backard'] = log

    # video_similarity = feat_t @ feat_v.T

    # log = compute_metric_ret_area((area - video_similarity), ids, ids_txt, direction='backward')
    # log = {k.replace('backward','area_video'): v for k,v in log.items()}
    # val_log[f'ret_area_back_with_video'] = log

    print("Conditional Features shape:{}".format(len(store_dict[f'condition_feats_{task}'])))
    print("Each Conditional Feature sample shape:{}".format(store_dict[f'condition_feats_{task}'][0].shape))


    print("conditional features length:", len(store_dict[f'condition_feats_v_not_reshaped']))
    print("Each conditional features sample shape:", store_dict[f'condition_feats_v_not_reshaped'][0].shape)

    store_dict[f'condition_feats_{task}'] = torch.cat(store_dict[f'condition_feats_{task}'],dim=0)
    store_dict[f'condition_feats_v_not_reshaped'] = torch.cat(store_dict[f'condition_feats_v_not_reshaped'],dim=0)
    itm_rerank_num = model.config.itm_rerank_num
    #itm_rerank_num = 30
    score_matrix = refine_score_matrix(store_dict['condition_feats_v_not_reshaped'], input_ids, attention_mask, -volume , model, itm_rerank_num, direction='forward', selected_frame_num=selected_frame_num)#-(area-video_similarity)
    log = compute_metric_ret(score_matrix, ids, ids_txt, direction='forward')
    log = {k.replace('forward','volume_ITM_T2D'): v for k,v in log.items()}


    score_matrix = refine_score_matrix(store_dict['condition_feats_v_not_reshaped'], input_ids, attention_mask, -volume, model, itm_rerank_num, direction='backward', selected_frame_num=selected_frame_num) #-(area-video_similarity)
    log2 = compute_metric_ret(score_matrix, ids, ids_txt, direction='backward')
    log2 = {k.replace('backward','volume_ITM_D2T'): v for k,v in log2.items()}
    log.update(log2)
    
    val_log[f'ret_itm_area'] = log
    
    
    cosine_TV = torch.matmul(feat_t, feat_v.permute(1,0))
    cosine_TV = compute_metric_ret(cosine_TV, ids, ids_txt, direction='forward')
    val_log[f'cosine_TV'] = cosine_TV
    
    cosine_VT = torch.matmul(feat_v, feat_t.permute(1,0))
    cosine_VT = compute_metric_ret(cosine_VT, ids, ids_txt, direction='forward')
    val_log[f'cosine_VT'] = cosine_VT
    
    cosine_TA = torch.matmul(feat_t, feat_a.permute(1,0))
    cosine_TA = compute_metric_ret(cosine_TA, ids, ids_txt, direction='forward')
    val_log[f'cosine_TA'] = cosine_TA
    
    cosine_AT = torch.matmul(feat_a, feat_t.permute(1,0))
    cosine_AT = compute_metric_ret(cosine_AT, ids, ids_txt, direction='forward')
    val_log[f'cosine_AT'] = cosine_AT
    
    ## compute itc_score
    #for task in subtasks:
    #    if  task == "tvas" or task == "tva":
    #        continue
    #    #store_dict[f'feat_cond_{task}'] =  torch.cat(store_dict[f'feat_cond_{task}'], dim = 0)
    #    #store_dict[f'feat_cond_{task}'] = ddp_allgather(store_dict[f'feat_cond_{task}'])
    #    if task=='tv':
    #        score_matrix_t_cond = torch.matmul(feat_t, feat_v.permute(1,0))
    #    elif task=='ta':
    #        score_matrix_t_cond = torch.matmul(feat_t, feat_a.permute(1,0))
    #    store_dict[f'score_matrix_t_cond_{task}'] = score_matrix_t_cond
    #    log = compute_metric_ret(score_matrix_t_cond, ids, ids_txt, direction='forward')
    #    log = {k.replace('forward','video'): v for k,v in log.items()}
    #    if model.config.ret_bidirection_evaluation:
    #        log2 = compute_metric_ret(score_matrix_t_cond, ids, ids_txt, direction='backward')
    #        log2 = {k.replace('backward','txt'): v for k,v in log2.items()}
    #        log.update(log2)
#
    #    val_log[f'ret_itc_{task}'] = log


    #### compute itm_score
    #for task in subtasks:
    #    if  task == "tvas" or task == "tva":
    #        continue
    #    if task!="tvas" and task!="tva":
    #        store_dict[f'condition_feats_{task}'] = torch.cat(store_dict[f'condition_feats_{task}'],dim=0)
    #    itm_rerank_num = model.config.itm_rerank_num
    #    score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task}'], input_ids, attention_mask, store_dict[f'score_matrix_t_cond_{task}'], model, itm_rerank_num, direction='forward')
    #    log = compute_metric_ret(score_matrix, ids, ids_txt, direction='forward')
    #    log = {k.replace('forward','video'): v for k,v in log.items()}
#
    #    if model.config.ret_bidirection_evaluation:
    #        score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task}'], input_ids, attention_mask, store_dict[f'score_matrix_t_cond_{task}'], model, itm_rerank_num, direction='backward')
    #        log2 = compute_metric_ret(score_matrix, ids, ids_txt, direction='backward')
    #        log2 = {k.replace('backward','txt'): v for k,v in log2.items()}
    #        log.update(log2)
#
    #    val_log[f'ret_itm_{task}'] = log

    if dist.get_rank() == 0:
        wandb.log(val_log)
        
    return val_log

def refine_score_matrix(condition_feats, input_ids, attention_mask, score_matrix_t_cond, model, itm_rerank_num, direction='forward',    selected_frame_num=None):
    """ Refine the initial retrieval score with cross-modal re-ranking
    """

    top_k = itm_rerank_num
    if direction=='forward':
        idxs = score_matrix_t_cond.topk(top_k,dim=1)[1]
    else:
        idxs = score_matrix_t_cond.topk(top_k,dim=0)[1]
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    nums = score_matrix_t_cond.shape[0]//world_size +1
    
    score_matrix_t_cond_new = torch.zeros_like(score_matrix_t_cond)
    idxs_new = torch.zeros_like(score_matrix_t_cond_new).long()



    if direction=='forward':
        for i in range(len(idxs)):
            for j in idxs[i]:
                idxs_new[i][j] = 1
    else:
        for i in range(idxs.shape[1]):
            for j in idxs[:,i]:
                idxs_new[j][i] = 1

    #print(idxs_new)
    cur_length = condition_feats.shape[0]
    length_ls = all_gather_list(cur_length)
    start = 0
    start_ls = []
    end_ls = []
    for l in range(len(length_ls)):
        start_ls.append(start)
        end_ls.append(start+length_ls[l])
        start = start+length_ls[l]
    
    cur_score_matrix_t_cond = score_matrix_t_cond[:,start_ls[rank]:end_ls[rank]]
    cur_score_matrix_t_cond_new = score_matrix_t_cond_new[:,start_ls[rank]:end_ls[rank]]
    cur_idxs_new = idxs_new[:,start_ls[rank]:end_ls[rank]]

    if dist.get_rank() == 0:
        pbar = tqdm(total=cur_length)
    else:
        pbar = NoOp()
    for i in range(cur_length):
        if sum(cur_idxs_new[:,i] == 1) == 0:
            continue
        cur_scores = []
        cur_input_ids = input_ids[(cur_idxs_new[:,i] == 1)]
        cur_attention_mask = attention_mask[(cur_idxs_new[:,i] == 1)]
        
        #print("cur_input_ids shape:", cur_input_ids.shape)
        #print("Current input ids:", cur_input_ids)
        #print("idx shape", idxs.shape)
        #print(f"idxs {idxs}")
        #print("(cur_idxs_new[:,i] == 1) = {}:".format(cur_idxs_new[:,i] == 1))

        #cur_condition_feats = condition_feats[i].unsqueeze(0).expand(cur_input_ids.shape[0],-1,-1)
        
        cur_condition_feats = []
        for j in range(cur_input_ids.shape[0]):
            if direction=='forward':
                selected_frames_cond = condition_feats[i][selected_frame_num[j][i]]
            else:
                selected_frames_cond = condition_feats[i][selected_frame_num[i][j]]
            selected_frames_cond = selected_frames_cond.view(-1, selected_frames_cond.size(-1)).unsqueeze(0)
            #print(selected_frames_cond.shape)
            #cur_condition_feats.append(condition_feats[i][selected_frame_num[i][j]].unsqueeze(0))
            cur_condition_feats.append(selected_frames_cond)
        cur_condition_feats = torch.cat(cur_condition_feats,dim=0)
        
        #print("cur_condition_feats shape:", cur_condition_feats.shape)
        #print("Current attention mask:", cur_attention_mask.shape)
        #print("Current input ids shape:", cur_input_ids.shape)
        total_len = len(cur_condition_feats)
        #print("total_len:", total_len)
        small_batch=25
        times = total_len//small_batch if total_len%small_batch==0 else total_len//small_batch+1

        for k in range(times):
            #print(k)
            slice_input_ids = cur_input_ids[k*small_batch:(k+1)*small_batch]
            slice_attention_mask = cur_attention_mask[k*small_batch:(k+1)*small_batch]
            slice_condition_feats = cur_condition_feats[k*small_batch:(k+1)*small_batch]
            #print("slice_condition_feats shape:", slice_condition_feats.shape)
            #print("slice_input_ids shape:", slice_input_ids.shape)
            #print("slice_attention_mask shape:", slice_attention_mask.shape)
            slice_scores = model.compute_slice_scores(slice_condition_feats, slice_input_ids, slice_attention_mask) 
            #print("slice_scores shape:", slice_scores.shape)
            cur_scores.append(slice_scores)

        #print("Number of score slices:", len(cur_scores))
        cur_scores = torch.cat(cur_scores,dim=0)

        #print("cur_scores shape:", cur_scores.shape)
        cur_score_matrix_t_cond_new[:,i][(cur_idxs_new[:,i] == 1)] = cur_scores
        pbar.update(1)
    pbar.close()
    
    score_matrix_t_cond = ddp_allgather(cur_score_matrix_t_cond_new.T.contiguous()).T

    return score_matrix_t_cond






def compute_metric_ret(score_matrix, ids, ids_txt, direction='forward'):



    assert score_matrix.shape == (len(ids_txt),len(ids))

    if direction == 'forward': ### text-to-vision retrieval
        indice_matrix = score_matrix.sort(dim=-1,descending=True)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            # gt_indice = ids.index(ids_txt[i][0])
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)
        v_medianR = torch.median(rank).item() +1
        v_meanR = torch.mean(rank).item() +1
 
        eval_log = {'forward_r1': round(vr_r1*100,1),
                    'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1)
                   }
   
    else: ### vision-to-text retrieval
       
        indice_matrix = score_matrix.sort(dim=0,descending=True)[1].permute(1,0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices=[]
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() +1
        t_meanR = torch.mean(rank).item() +1

        eval_log = {
                    'backward_r1': round(tr_r1*100,1),
                    'backward_recall': f'{round(tr_r1*100,1)}/{round(tr_r5*100,1)}/{round(tr_r10*100,1)}',
                    'backward_ravg': round((tr_r1 + tr_r5 + tr_r10)/3 *100,1)
                  }
    

    return eval_log


def compute_metric_ret_area(score_matrix, ids, ids_txt, direction='forward'):



    assert score_matrix.shape == (len(ids_txt),len(ids))

    if direction == 'forward': ### text-to-vision retrieval
        indice_matrix = score_matrix.sort(dim=-1,descending=False)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            # gt_indice = ids.index(ids_txt[i][0])
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)
        v_medianR = torch.median(rank).item() +1
        v_meanR = torch.mean(rank).item() +1
 
        eval_log = {'forward_r1': round(vr_r1*100,1),
                    'forward_recall': f'{round(vr_r1*100,1)}/{round(vr_r5*100,1)}/{round(vr_r10*100,1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10)/3 *100,1)
                   }
   
    else: ### vision-to-text retrieval
       
        indice_matrix = score_matrix.sort(dim=0,descending=False)[1].permute(1,0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices=[]
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))
        
        rank = torch.tensor(rank).to(score_matrix)
        
        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() +1
        t_meanR = torch.mean(rank).item() +1

        eval_log = {
                    'backward_r1': round(tr_r1*100,1),
                    'backward_recall': f'{round(tr_r1*100,1)}/{round(tr_r5*100,1)}/{round(tr_r10*100,1)}',
                    'backward_ravg': round((tr_r1 + tr_r5 + tr_r10)/3 *100,1)
                  }
    

    return eval_log




def compute_metric_cap(results, annfile_path, process=True):
    coco = COCO(annfile_path)
    cocoRes = coco.loadRes(results)
    cocoEval = COCOEvalCap(coco, cocoRes, process)
    cocoEval.evaluate()
    metric = cocoEval.eval
    metric = {k: round(v*100,2)  for k,v in metric.items()}
    return metric
