import h5py
import numpy as np
import torch
import sys
from config import get_config
from dataset import create_dataloader
from evaluation_metrics import get_corr_coeff
from generate_summary import generate_summary
from model import set_model
from utils import report_params,get_gt


def ap_scoress(score, label, k=5, **kwargs):
    label = torch.tensor(label)
    inds = torch.argsort(score, descending=True)
    label = torch.where(label > label.median(), 1.0, .0)
    label = label[inds].tolist()[:k]

    if (num_gt := sum(label)) == 0:
        return 0, 0, 0

    hits = ap = rec = 0
    prc = 1

    for j, gt in enumerate(label):
        hits += gt

        _rec = hits / num_gt
        _prc = hits / (j + 1)

        ap += (_rec - rec) * (prc + _prc) / 2
        rec, prc = _rec, _prc

    return ap, rec, prc

def ap_score(pred_imp_scores, videos, user_scores, k=5):
    collected = []
    for pred_imp_score,video in zip(pred_imp_scores,videos):
        pred_imp_score = np.squeeze(pred_imp_score).tolist()
        user = int(video.split("_")[-1])
        video_ap = []
    
        curr_user_score = user_scores[user-1]
        for annotation in range(len(curr_user_score)):
            label = torch.tensor(curr_user_score[annotation].tolist())
            inds = torch.argsort(torch.tensor(pred_imp_score), descending=True)
            label = torch.where(label > label.median(), 1.0, .0)
            label = label[inds].tolist()[:k]

            if (num_gt := sum(label)) == 0:
                video_ap.append(0)
                continue

            hits = ap = rec = 0
            prc = 1

            for j, gt in enumerate(label):
                hits += gt

                _rec = hits / num_gt
                _prc = hits / (j + 1)

                ap += (_rec - rec) * (prc + _prc) / 2
                rec, prc = _rec, _prc

            video_ap.append(ap)

        collected.append(sum(video_ap) / len(video_ap))

    mean_ap = sum(collected) / len(collected)
    results = dict(mAP=round(mean_ap, 5))

    return mean_ap

    return ap, rec, prc

def experimento(exp):
    # Load configurations
    config = get_config()

    # Print the number of parameters
    report_params(
        model_name=config.model_name,
        Scale=config.Scale,
        Softmax_axis=config.Softmax_axis,
        Balance=config.Balance,
        Positional_encoding=config.Positional_encoding,
        Positional_encoding_shape=config.Positional_encoding_shape,
        Positional_encoding_way=config.Positional_encoding_way,
        Dropout_on=config.Dropout_on,
        Dropout_ratio=config.Dropout_ratio,
        Classifier_on=config.Classifier_on,
        CLS_on=config.CLS_on,
        CLS_mix=config.CLS_mix,
        key_value_emb=config.key_value_emb,
        Skip_connection=config.Skip_connection,
        Layernorm=config.Layernorm
    )

    # Start testing
    for dataset in config.datasets:
        user_scores = get_gt(dataset)
        split_kendalls = []
        split_spears = []
        split_maps = []
        for split_id,(train_loader,test_loader) in enumerate(create_dataloader(dataset)):
            model = set_model(
                model_name=config.model_name,
                Scale=config.Scale,
                Softmax_axis=config.Softmax_axis,
                Balance=config.Balance,
                Positional_encoding=config.Positional_encoding,
                Positional_encoding_shape=config.Positional_encoding_shape,
                Positional_encoding_way=config.Positional_encoding_way,
                Dropout_on=config.Dropout_on,
                Dropout_ratio=config.Dropout_ratio,
                Classifier_on=config.Classifier_on,
                CLS_on=config.CLS_on,
                CLS_mix=config.CLS_mix,
                key_value_emb=config.key_value_emb,
                Skip_connection=config.Skip_connection,
                Layernorm=config.Layernorm
            )
            model.load_state_dict(torch.load(f'./weights/{exp}/{dataset}/split{split_id+1}.pt', map_location='cpu'))
            model.to(config.device)
            model.eval()

            kendalls = []
            spears = []
            map_scores = []
            with torch.no_grad():
                for feature,_,dataset_name,video_num, pascore in test_loader:
                    feature = feature.to(config.device)
                    output = model(feature)
                    if exp == 'hlpa':
                        output = (output+pascore)/2
                    with h5py.File(f'./data/eccv16_dataset_{dataset_name.lower()}_google_pool5.h5','r') as hdf:
                        if "HiSum" in dataset_name:
                            user_summary = np.array(hdf[video_num]['gtsummary'][()]).astype(np.float32)
                        else:
                            user_summary = np.array(hdf[video_num]['user_summary'])
                        sb = np.array(hdf[f"{video_num}/change_points"])
                        n_frames = np.array(hdf[f"{video_num}/n_frames"])
                        positions = np.array(hdf[f"{video_num}/picks"])
                    scores = output.squeeze().clone().detach().cpu().numpy().tolist()
                    summary = generate_summary([sb], [scores], [n_frames], [positions])[0]

                    if dataset_name=='SumMe':
                        spear,kendall = get_corr_coeff([summary],[video_num],dataset_name,user_summary)
                    elif dataset_name=='TVSum':
                        spear,kendall = get_corr_coeff([scores],[video_num],dataset_name,user_scores)
                        ap = ap_score([scores],[video_num],user_scores)
                        map_scores.append(ap)
                    elif 'HiSum' in dataset_name:
                        spear,kendall = get_corr_coeff([summary],[video_num],dataset_name,user_summary)                
                    spears.append(spear)
                    kendalls.append(kendall)
            split_kendalls.append(np.mean(kendalls))
            split_spears.append(np.mean(spears))
            split_maps.append(np.mean(map_scores))
            print("[Split{}]Kendall:{:.3f}, Spear:{:.3f}".format(
                split_id,split_kendalls[split_id],split_spears[split_id]
            ))
            if dataset == 'TVSum':
                print(f"mAP: {map_scores}")

        if dataset == 'TVSum':
            print("[FINAL - {}]Kendall:{:.3f}, Spear:{:.3f}, mAP@5: {:.3f}".format(
                dataset,np.mean(split_kendalls),np.mean(split_spears), np.mean(split_maps)
            ))
        else:
            print("[FINAL - {}]Kendall:{:.3f}, Spear:{:.3f}".format(
                dataset,np.mean(split_kendalls),np.mean(split_spears)
            ))


if __name__ == '__main__':
    exps = ['hl', 'hlpa', 'hlunpa']
    datasets = ['TVSum']
    for exp in exps:
        experimento(exp)