import os
import gc
import copy
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from sklearn.metrics import precision_score, roc_curve

from scripts.shimmer_data_loader import load_stroke_demographics
from scripts.shimmer_data_loader import STROKE_SENSOR_LOCATION, SHIMMER_SENSOR_LOCATION
import scripts.kinematic_approach as kinematic_approach
import scripts.features as mvmt_elem_features
# import scripts.sliding_window as sliding_window

sensor_offset = {
    1: 367,
    2: 167,
    3: 167,
    4: 0,
    5: 367,
    7: 400,
    9: 167,
    10: 133000,
    11: 0,
    12: 20754,
    13: 0,
    14: 121855,
    16: 367,
    17: 34,
    18: 0,
    19: 0,
    20: 116
}

filename = os.path.join('input', 'stroke_subject_info.xlsx')
demographics = load_stroke_demographics(filename)
del filename
    

def orient_estimation(stroke_data, ahrs_filter_type, output_dir):
    for subj in stroke_data:
        for loc in STROKE_SENSOR_LOCATION:
            if loc in dir(stroke_data[subj]):
                # Extract data
                data_obj = getattr(stroke_data[subj], loc)

                # Calculate rotation matrix
                kinematic_approach.fuse_and_rotate(data_obj, ahrs_filter_type)

                # Save
                setattr(stroke_data[subj], loc, data_obj)
                del loc, data_obj
                gc.collect()
        del subj

    # Save
    filename = 'stroke_data_%s.npz' % ahrs_filter_type
    np.savez(os.path.join(output_dir, filename), **{'data': stroke_data})


def rotation(stroke_data, num_devices, ahrs_filter_type):
    for subj in stroke_data:
        for loc in SHIMMER_SENSOR_LOCATION:
            if loc in dir(stroke_data[subj]):
                # Extract data
                data_obj = getattr(stroke_data[subj], loc)

                # Rotation
                if num_devices == 1:
                    trunk_obj = None
                elif num_devices == 2:
                    trunk_obj = copy.deepcopy(stroke_data[subj].trunk)
                kinematic_approach.rotation_wrist(data_obj, trunk_obj, ahrs_filter_type)

                # Save
                setattr(stroke_data[subj], loc, data_obj)
                del loc, data_obj, trunk_obj
                gc.collect()
        del subj
    return stroke_data


# def bandpass_filter_gravity(stroke_data, order, fc_lp, fc_hp):
#     for subj in stroke_data:
#         for loc in SHIMMER_SENSOR_LOCATION:
#             if loc in dir(stroke_data[subj]):
#                 # Extract data
#                 data_obj = getattr(stroke_data[subj], loc)

#                 # Filter data
#                 kinematic_approach.bandpass_filter_gravity(data_obj, order=order, fc_lp=fc_lp, fc_hp=fc_hp)

#                 # Save
#                 setattr(stroke_data[subj], loc, data_obj)
#                 del loc, data_obj
#                 gc.collect()
#         del subj
#     return stroke_data


def trim_data(stroke_data, annotations, offsets):
    for subj in stroke_data:
        # Extract annotation
        annot = annotations[subj].copy()
        annot_mask = annot['Tier'] == 'left_wrist'
        annot = annot[annot_mask]
        del annot_mask

        # Define start_time and end_time
        start_time = 0
        end_time = annot['End'].iloc[-1]

        # Load offset
        offset = sensor_offset[subj]

        for loc in SHIMMER_SENSOR_LOCATION:
            if loc in dir(stroke_data[subj]):
                # Extract data
                data_obj = getattr(stroke_data[subj], loc)
                
                # Trim data
                kinematic_approach.trim_stroke_data_by_time(data_obj, offset, start_time, end_time)
                
                # Save data
                setattr(stroke_data[subj], loc, data_obj)
                del loc, data_obj
        
        del subj, annot, start_time, end_time, offset
    return stroke_data


def filtering(stroke_data, filter_type='lowpass'):
    for subj in stroke_data:
        for loc in SHIMMER_SENSOR_LOCATION:
            if loc in dir(stroke_data[subj]):
                # Extract data
                data_obj = getattr(stroke_data[subj], loc)
                
                # Filter and integrate
                kinematic_approach.filter_and_integrate(data_obj, filter_type=filter_type)
                
                # Save data
                setattr(stroke_data[subj], loc, data_obj)
                del loc, data_obj
    return stroke_data


def segmentation(stroke_data, annotations, method, delta, args, p2p_annotations=None, **kwargs):
    results = []
    for subj in stroke_data:
        for loc in SHIMMER_SENSOR_LOCATION:
            if loc in dir(stroke_data[subj]):
                # Extract data
                data_obj = getattr(stroke_data[subj], loc)
                
                # Extract annotation
                annot = annotations[subj].copy()
                annot_mask = (annot['Tier'] == loc)
                if method in ['all_mvmt_elem', 'sliding_window']:
                    annot_mask &= (annot['Label'].isin(['discard']))
                elif method in ['p2p_mvmt_elem', 'non_p2p_mvmt_elem', 'p2p_true', 'p2p_pred']:
                    annot_mask &= (annot['Label'].isin(['reach', 'reposition', 'transport']))
                else:
                    raise ValueError('Incorrect segmentation method.')
                
                # if method in ['all_mvmt_elem', 'p2p_mvmt_elem', 'non_p2p_mvmt_elem', 'sliding_window']:
                #     annot_mask &= (annot['Label'].isin(['discard']))
                # elif method in ['p2p_true', 'p2p_pred']:
                #     annot_mask &= (annot['Label'].isin(['reach', 'reposition', 'transport']))
                # else:
                #     raise ValueError('Incorrect segmentation method.')
                    
                annot = annot[annot_mask]
                del annot_mask
                  
                # Segmentation
                if method == 'all_mvmt_elem':
                    data_obj.start_idx = 0
                    res = kinematic_approach.decompose_all_mvmt_elem(data_obj, annot)
                    results.extend(res)
                    del res
                elif method == 'p2p_mvmt_elem':
                    data_obj.start_idx = 0
                    res = kinematic_approach.decompose_p2p_mvmt_elem(data_obj, annot)
                    results.extend(res)
                    del res
                    
#                     # extract mvmt segments
#                     data_obj.start_idx = 0
#                     res = kinematic_approach.decompose_all_mvmt_elem(data_obj, annot)

#                     # extract p2p annotations
#                     p2p_annot = p2p_annotations[subj].copy()
#                     p2p_annot_mask = (p2p_annot['Tier'] == loc)
#                     p2p_annot = p2p_annot[p2p_annot_mask]
#                     del p2p_annot_mask
                    
#                     # determine if a segment is p2p or not
#                     p2p_mask = []
#                     for i, r in res:
#                         start, end = r.time[0], r.time[-1]
                        
#                         pmask = (end < p2p_annot['Begin']) | (start > p2p_annot['End'])
#                         if pmask.all():
#                             p2p_mask.append(False)
#                         else:
#                             p2p_mask.append(True)
                            
#                             # # Assign p2p_id
#                             # r.p2p_id = p2p_annot[~pmask].index[0]
#                             # res[i] = r
#                     p2p_mask = np.array(p2p_mask)
                    
#                     # save
#                     res = res[p2p_mask]
#                     results.extend(res)
#                     del res, p2p_mask
                elif method == 'non_p2p_mvmt_elem':
                    data_obj.start_idx = 0
                    res = kinematic_approach.decompose_non_p2p_mvmt_elem(data_obj, annot)
                    results.extend(res)
                    del res
                    
#                     # extract mvmt segments
#                     data_obj.start_idx = 0
#                     res = kinematic_approach.decompose_all_mvmt_elem(data_obj, annot)
      
#                     # extract p2p annotations
#                     p2p_annot = p2p_annotations[subj].copy()
#                     p2p_annot_mask = (p2p_annot['Tier'] == loc)
#                     p2p_annot = p2p_annot[p2p_annot_mask]
#                     del p2p_annot_mask
                    
#                     # determine if a segment is p2p or not
#                     non_p2p_mask = []
#                     for r in res:
#                         start, end = r.time[0], r.time[-1]
                        
#                         if ((end < p2p_annot['Begin']) | (start > p2p_annot['End'])).all():
#                             non_p2p_mask.append(True)
#                         else:
#                             non_p2p_mask.append(False)
#                     non_p2p_mask = np.array(non_p2p_mask)
                    
#                     # save
#                     res = res[non_p2p_mask]
#                     results.extend(res)
#                     del res, non_p2p_mask
                elif method in ['p2p_pred', 'p2p_true']:
                    res = kinematic_approach.decompose_p2p_wnd(data_obj, annot, delta=delta, **kwargs)
                    results.extend(res)
                    del res
                elif method == 'sliding_window':
                    res = sliding_window.sliding_windows(data_obj, annot)
                    results.extend(res)
                    del res
                else:
                    raise ValueError('Invalid seg. method')
    results = np.array(results)
    return results


def nan_padded_matrix(arrs):
    height = len(arrs)
    max_width = max([a.shape[0] for a in arrs])
    result = np.full((height, max_width), np.nan, dtype='float64')
    for i in range(height):
        width = arrs[i].shape[0]
        result[i, 0:width] = arrs[i]
    return result


def overlap(subjs, locs, axes, start_indices, end_indices):
    discard = np.zeros_like(subjs, dtype=bool)
    
    for subj, loc, axis in set(zip(subjs, locs, axes)):
        mask = (subjs == subj)
        mask &= (locs == loc)
        mask &= (axes == axis)
        
        start_idx = start_indices[mask]
        end_idx = end_indices[mask]
        
        start_idx_update = []
        end_idx_update = []
        
        subj_discard = []
        for i in range(len(start_idx)):
            start, end = start_idx[i], end_idx[i]
            begin_ = np.maximum(start_idx_update, start)
            end_ = np.minimum(end_idx_update, end)
            
            overlap = (end_ - begin_) / (end - start)
            if np.all(overlap <= 0):
                start_idx_update.append(start)
                end_idx_update.append(end)
                subj_discard.append(False)
            else:
                subj_discard.append(True)
            del start, end
            del begin_, end_
            del overlap
        
        del start_idx, end_idx
        del start_idx_update, end_idx_update
        discard[mask] = subj_discard
        del mask, subj_discard
    return discard


def organize_mvmt_elems(stroke_elements):
    subjs = []
    locs = []
    axes = []
    aff_sides = []
    dur = []
    vel = []
    norm_vel = []
    acc = []
    jerk = []
    disp = []
    discard = []
    start_indices = []
    end_indices = []
    p2p_id = []

    for element in stroke_elements:
        subjs.append(element.subject)
        locs.append(element.location)
        axes.append(element.axis)
        aff_sides.append(demographics[element.subject].impaired_side in element.location)
        dur.append(element.dur)
        vel.append(element.vel)
        norm_vel.append(element.norm_vel)
        acc.append(element.acc)
        jerk.append(element.jerk)
        disp.append(element.disp)
        discard.append(element.discard)
        start_indices.append(element.start_idx)
        end_indices.append(element.end_idx)
        p2p_id.append(element.p2p_id)
    
    subjs = np.array(subjs)
    locs = np.array(locs)
    axes = np.array(axes)
    aff_sides = np.array(aff_sides)
    dur = np.array(dur)
    vel = nan_padded_matrix(vel)
    norm_vel = np.array(norm_vel)
    acc = nan_padded_matrix(acc)
    jerk = nan_padded_matrix(jerk)
    disp = np.array(disp)
    discard = np.array(discard)
    start_indices = np.array(start_indices)
    end_indices = np.array(end_indices)
    p2p_id = np.array(p2p_id)
    
    # discard mvmt elems
    subjs = subjs[~discard]
    locs = locs[~discard]
    axes = axes[~discard]
    aff_sides = aff_sides[~discard]
    dur = dur[~discard]
    vel = vel[~discard]
    norm_vel = norm_vel[~discard]
    acc = acc[~discard]
    jerk = jerk[~discard]
    disp = disp[~discard]
    start_indices = start_indices[~discard]
    end_indices = end_indices[~discard]
    p2p_id = p2p_id[~discard]
    
    # overlap
    exclusion_mask = overlap(subjs, locs, axes, start_indices, end_indices)
    subjs = subjs[~exclusion_mask]
    locs = locs[~exclusion_mask]
    axes = axes[~exclusion_mask]
    aff_sides = aff_sides[~exclusion_mask]
    dur = dur[~exclusion_mask]
    vel = vel[~exclusion_mask]
    norm_vel = norm_vel[~exclusion_mask]
    acc = acc[~exclusion_mask]
    jerk = jerk[~exclusion_mask]
    disp = disp[~exclusion_mask]
    start_indices = start_indices[~exclusion_mask]
    end_indices = end_indices[~exclusion_mask]
    p2p_id = p2p_id[~exclusion_mask]
    
    return subjs, locs, axes, aff_sides, dur, vel, norm_vel, acc, jerk, disp, start_indices, end_indices, p2p_id


def remove_invalid_features(features, feature_names):
    invalid_mask = np.any(np.isnan(features), axis=0) | np.any(np.isinf(features), axis=0)
    features = features[:, ~invalid_mask]
    feature_names = feature_names[~invalid_mask]
    # print(f'Removed {invalid_mask.sum()} features.')
    # print(f'Extracted {(~invalid_mask).sum()} features.')
    return features, feature_names


def split_data(subjs, locs, axes=[], split_method=1, stroke_data=None, start_indices=None, end_indices=None, p2p_id=None, annotations=None):
    '''
    subjs: subjects
    locs locations
    start_indices: start_idx of mvmt elem
    '''
    if len(axes) == 0:
        axes = np.array([0 for _ in range(len(subjs))])
    print('split method: ', split_method, flush=True)
    indices_1st = []; indices_2nd = []
    for uniq_s, uniq_l, uniq_a in sorted(set(zip(subjs, locs, axes))):
        # Extract data from that limb
        mask = subjs == uniq_s
        mask &= (locs == uniq_l)
        mask &= (axes == uniq_a)
        idx = np.where(mask)[0]

        if split_method == 1:
            # 1) the first half of time-series was put in the first set
            # and the second half of time-series was put in the second set
            threshold = getattr(stroke_data[uniq_s], uniq_l).local.acc.shape[0] // 2
            indices_1st.extend(idx[start_indices[mask] < threshold])
            indices_2nd.extend(idx[start_indices[mask] >= threshold])
        elif split_method == 1.5:
            # 1.5) split_method 1 adapted for p2p_id
            threshold = getattr(stroke_data[uniq_s], uniq_l).local.acc.shape[0] // 2
            for p2p in np.unique(p2p_id[mask]):
                if start_indices[mask][(p2p_id[mask] == p2p)][0] < threshold:
                    indices_1st.extend(idx[p2p_id[mask] == p2p])
                else:
                    indices_2nd.extend(idx[p2p_id[mask] == p2p])
#         elif split_method == 2:
#             # 2) the 1st mvmt elem was put in the first set
#             # the 2nd mvmt elem was put in the second set
#             # the 3rd mvmt elem was put in the first set
#             indices_1st.extend(idx[::2])
#             indices_2nd.extend(idx[1::2])
#         elif split_method == 3:
#             # 3) mvmt elems in the 1st reaching were put in the first set
#             # the 2nd mvmt elems were put in the second set
#             # the 3rd mvmt elems were put in the first set
#             for n, p2p in enumerate(np.unique(p2p_id[mask])):
#                 if n % 2 == 0:
#                     indices_1st.extend(idx[p2p_id[mask] == p2p])
#                 else:
#                     indices_2nd.extend(idx[p2p_id[mask] == p2p])
#         elif split_method == 3.5:
#             # 3.5) split p2p_id into halves
#             n_p2p = len(np.unique(p2p_id[mask]))
#             for n, p2p in enumerate(np.unique(p2p_id[mask])):
#                 if n < n_p2p // 2:
#                     indices_1st.extend(idx[p2p_id[mask] == p2p])
#                 else:
#                     indices_2nd.extend(idx[p2p_id[mask] == p2p])
#         elif split_method == 4:
#             # 4) random split
#             np.random.seed(42)
#             m = np.zeros_like(idx, dtype=bool)
#             n = np.random.choice(np.arange(len(idx), dtype=int), size=(len(idx) // 2, ), replace=False)
#             m[n] = True
#             indices_1st.extend(idx[m])
#             indices_2nd.extend(idx[~m])
#         elif split_method == 5:
#             # 5) mvmt elems in the 1st x~min were put in the first set
#             # mvmt elems in the 2nd x~min were put in the second set
#             sample_len = getattr(stroke_data[uniq_s], uniq_l).local.acc.shape[0]
#             stp_sz = 50 * 60 * 5
#             for i, start in enumerate(np.arange(0, sample_len, stp_sz)):
#                 end = start + stp_sz
#                 m = (start_indices[mask] > start) & (start_indices[mask] <= end)
#                 if i % 2 == 0:
#                     indices_1st.extend(idx[m])
#                 else:
#                     indices_2nd.extend(idx[m])
#                 del end, m
#         elif split_method == 6:
#             # 6) split data based on annotations
#             annot = annotations[uniq_s].copy()
#             annot_mask = annot['Tier'] == 'activity'
#             annot = annot[annot_mask]
#             del annot_mask
            
#             for _, (tier, begin, end, label) in annot.iterrows():
#                 annot_start = begin * 50
#                 annot_end = end * 50
#                 annot_mid = (annot_start + annot_end) / 2
#                 indices_1st.extend(idx[(start_indices[mask] > annot_start) & (start_indices[mask] <= annot_mid)])
#                 indices_2nd.extend(idx[(start_indices[mask] > annot_mid) & (start_indices[mask] <= annot_end)])
#         elif split_method == 7:
#             # 7) within each p2p mvmt
#             # split the mvmt elem based on odd/even
#             for n, p2p in enumerate(np.unique(p2p_id[mask])):
#                 p2p_mask = p2p_id[mask] == p2p
#                 indices_1st.extend(idx[(idx % 2 == 0) & p2p_mask])
#                 indices_2nd.extend(idx[(idx % 2 == 1) & p2p_mask])
#         elif split_method == 8:
#             # method 5 tailored for p2p_id
#             pass
            
    return indices_1st, indices_2nd


def reliability_test(features, features_1st, features_2nd, feature_names, 
                     feature_names_1st, feature_names_2nd, score_dict, args):
    # Reliability Test
    datadict = defaultdict(list)
    for i in range(feature_names.shape[0]):
        f = feature_names[i]
        datadict['feature_names'].append(f)

        # calculate correlation between
        # features and correlation score(s)
        if isinstance(score_dict, np.ndarray):
            corr = args.rfunc(features[:, i], score_dict)[0]
        elif isinstance(score_dict, dict):
            if args.corr_target == 'FM':
                corr = args.rfunc(features[:, i], score_dict['FM'])[0]
            elif args.corr_target == 'objective':
                corr = 0
                for score_key in score_dict:
                    if 'MAL' not in score_key:
                        corr += np.abs(args.rfunc(features[:, i], score_dict[score_key])[0])
                corr /= 3
            elif args.corr_target == 'all':
                corr = 0
                for score_key in score_dict:
                    corr += np.abs(args.rfunc(features[:, i], score_dict[score_key])[0])
                corr /= 5
        datadict['corr'].append(corr)
                                                
        # icc
        m = np.where(feature_names_1st == f)[0]
        n = np.where(feature_names_2nd == f)[0]
        if len(m) == 0 or len(n) == 0:
            datadict['icc'].append(np.nan)
        else:
            M = np.hstack((features_1st[:, m], features_2nd[:, n]))
            r, lb, ub, f, df1, df2, p = mvmt_elem_features.icc(M, args.icc_type)
            datadict['icc'].append(r)
    dataframe = pd.DataFrame(data=datadict)
    return dataframe

################

def summarize_output(filepath, subject_list):
    nn_results = defaultdict(list)
    for i, s in enumerate(np.unique(subject_list)):
        # Load output
        filename = '%02d.npz' % (s)
        if filename not in os.listdir(filepath):
            print('%s not in the folder.' % filename, flush=True)
            continue
        
        # Load data
        output = np.load(os.path.join(filepath, filename), allow_pickle=True)
        
        # Save affected mask and activity mask
        affected = output['affected']
        nn_results['affected'].append(affected)
        
        activity_mask = output['activity_mask']
        nn_results['activity_mask'].append(activity_mask)
        
        # Training set
        train_mask = output['train_mask']
        train_true = output['train_true']
        train_scores = output['train_scores']
        train_pred = np.argmax(train_scores, axis=1)
        
        nn_results['train_masks'].append(train_mask)
        nn_results['train_true'].append(train_true)
        nn_results['train_scores'].append(train_scores)
        nn_results['train_pred'].append(train_pred)
        del train_mask, train_true, train_scores, train_pred

        # Validation set
        val_mask = output['val_mask']
        val_true = output['val_true']
        val_scores = output['val_scores']
        val_pred = np.argmax(val_scores, axis=1)
        
        nn_results['val_masks'].append(val_mask)
        nn_results['val_true'].append(val_true)
        nn_results['val_scores'].append(val_scores)
        nn_results['val_pred'].append(val_pred)
        del val_mask, val_true, val_scores, val_pred

        # Test set
        # Test set contains both affected and unaffected limbs
        # Test set also contains non active data
        # When evaluting the final detection performance, 
        # we should look at everything in the test set
        # including non-active ones
        test_mask = output['test_mask']
        test_true = output['test_true']
        #########################################################################
        # test_scores = output['test_scores']
        test_scores = np.load(f'./input/predictions2_wnd_sz_3_overlap_1_3/{s:02d}.npz',
                              allow_pickle=True)['out']
        #########################################################################
        test_pred = np.argmax(test_scores, axis=1)
        
        nn_results['test_masks'].append(test_mask)
        nn_results['test_true'].append(test_true)
        nn_results['test_scores'].append(test_scores)
        nn_results['test_pred'].append(test_pred)
        del test_mask, test_true, test_scores, test_pred

        # Intermediate results
        nn_results['subjects'].append(s)
        nn_results['best_params'].append(output['best_params'])
        nn_results['train_loss_history'].append(output['train_loss_history'])
        nn_results['val_loss_history'].append(output['val_loss_history'])
        nn_results['train_score_history'].append(output['train_score_history'])
        nn_results['val_score_history'].append(output['val_score_history'])
        
        del output, activity_mask
    return nn_results


def helper(true, scores, target_score):
    fpr, tpr, thresholds = roc_curve(true, scores[:, 1])
    
    precision_scores = []
    for th in thresholds:
        pred = np.zeros_like(true)
        pred[scores[:, 1] >= th] = 1
        precision_scores.append(precision_score(true, pred))
        del pred
    precision_scores = np.array(precision_scores)
    
    best_idx = np.argmin(np.abs(precision_scores - target_score))
    return thresholds[best_idx]
    
    
def tuning_op(nn_results, target_score, seg_method='p2p_pred'):
    nn_results = copy.deepcopy(nn_results)
    for i in range(len(nn_results['subjects'])):
        # Extract data
        val_true = nn_results['val_true'][i]
        val_pred = nn_results['val_pred'][i]
        val_scores = nn_results['val_scores'][i]
        
        test_true = nn_results['test_true'][i]
        test_pred = nn_results['test_pred'][i]
        test_scores = nn_results['test_scores'][i]

        # Tuning op
        if seg_method == 'p2p_true':
            update_test_pred = test_true.copy()
        elif seg_method == 'p2p_pred':
            if target_score == -1:
                update_test_pred = test_pred.copy()
            else:
                if precision_score(val_true, val_pred) < target_score:
                    th = helper(val_true, val_scores, target_score)

                    # Apply the threshold on test set
                    update_test_pred = np.zeros_like(test_true)
                    update_test_pred[test_scores[:, 1] >= th] = 1
                else:
                    update_test_pred = test_pred.copy()
        # Save
        nn_results['update_test_pred'].append(update_test_pred)
        del update_test_pred
    return nn_results


def load_stroke_windows(path):
    windows = []
    for filename in sorted(os.listdir(path)):
        # Load data
        data = np.load(os.path.join(path, filename), allow_pickle=True)
        wnds = data['windows']
        del filename, data
        
        # Save
        windows.extend(wnds)
    return np.array(windows)


def organize_predictions(windows, nn_results):
    windows = copy.deepcopy(windows)
    
    n_windows = len(windows)
    
    wnd_subject = np.array([w.subject for w in windows])
    wnd_location = np.array([w.location for w in windows])
    
    wnd_true = np.zeros((n_windows, ))
    wnd_pred = np.zeros((n_windows, ))
    
    for i, s in enumerate(nn_results['subjects']):
        # Extract data
        y_true = nn_results['test_true'][i]
        y_pred = nn_results['update_test_pred'][i]
        
        # Extract subj data from windows
        subj_mask = wnd_subject == s
        
        # Save
        wnd_true[subj_mask] = y_true
        wnd_pred[subj_mask] = y_pred
        
        del y_true, y_pred, subj_mask
        gc.collect()
    
    return wnd_true, wnd_pred


def combine_windows(windows, labels):
    windows = copy.deepcopy(windows)
    
    subjects = np.array([w.subject for w in windows])
    locations = np.array([w.location for w in windows])
    start_time = np.array([w.time[0] for w in windows])
    end_time = np.array([w.time[-1] for w in windows])
    
    intervals = {}
    for s in np.unique(subjects):
        for l in np.unique(locations):
            key = (s, l)
            intervals[(s, l)] = []
            
            mask = (subjects == s)
            mask &= (locations == l)
            mask &= (labels == 1)
            running_start, running_end = np.inf, -np.inf
            for start, end in zip(start_time[mask], end_time[mask]):
                if start <= running_end:
                    running_end = end
                else:
                    intervals[key].append([running_start, running_end])
                    running_start = start
                    running_end = end
            intervals[key].append([running_start, running_end])
            intervals[key].pop(0)
            intervals[key] = np.array(intervals[key])
            if len(intervals[key]) == 0:
                del intervals[key]
    return intervals


def convert_pred_to_annotation(pred):
    results = defaultdict(lambda: defaultdict(list))
    
    for subj, loc in pred:
        for start_time, end_time in pred[subj, loc]:
            results[subj]['Tier'].append(loc)
            results[subj]['Begin'].append(start_time)
            results[subj]['End'].append(end_time)
            results[subj]['Label'].append('reach')
    
    annotations = dict()
    for subj in results:
        annotations[subj] = pd.DataFrame(data=results[subj])
    return annotations
            

# def segmentation_v2(stroke_data, post_pred, delta=0.0):
#     results = []
#     for subj in stroke_data:
#         for loc in SHIMMER_SENSOR_LOCATION:
#             if loc in dir(stroke_data[subj]):
#                 # Extract data
#                 data_obj = getattr(stroke_data[subj], loc)
                
#                 # Extract annotation
#                 n_samples = len(post_pred[subj, loc])
#                 annot = defaultdict(list)
#                 for i in range(n_samples):
#                     annot['Tier'].append(loc)
#                     annot['Begin'].append(post_pred[subj, loc][i][0])
#                     annot['End'].append(post_pred[subj, loc][i][1])
#                     annot['Label'].append('p2p')
   
#                 annot = pd.DataFrame(data=annot)

#                 # Segmentation
#                 res = kinematic_approach.decompose_p2p_mvmt_elem(data_obj, annot, delta=delta)
#                 results.extend(res)
#                 del res
        
#     results = np.array(results)
#     # print('{:d} movement elements were extracted.'.format(len(results)))
#     return results


# def organize_mvmt_elems_helper(stroke_elements):
#     subjs = []
#     locs = []
#     axes = []
#     aff_sides = []
#     dur = []
#     vel = []
#     norm_vel = []
#     acc = []
#     jerk = []
#     disp = []
#     discard = []
#     start_indices = []
#     end_indices = []
#     # p2p_id = []

#     for element in stroke_elements:
#         subjs.append(element.subject)
#         locs.append(element.location)
#         axes.append(element.axis)
#         aff_sides.append(demographics[element.subject].impaired_side in element.location)
#         dur.append(element.dur)
#         vel.append(element.vel)
#         norm_vel.append(element.norm_vel)
#         acc.append(element.acc)
#         jerk.append(element.jerk)
#         disp.append(element.disp)
#         discard.append(element.discard)
#         start_indices.append(element.start_idx)
#         end_indices.append(element.end_idx)
#         # if 'p2p_id' not in dir(element):
#         #     p2p_id.append(None)
#         # else:
#         #     p2p_id.append(element.p2p_id)

#     subjs = np.array(subjs)
#     locs = np.array(locs)
#     axes = np.array(axes)
#     aff_sides = np.array(aff_sides)
#     dur = np.array(dur)
#     vel = nan_padded_matrix(vel)
#     norm_vel = np.array(norm_vel)
#     acc = nan_padded_matrix(acc)
#     jerk = nan_padded_matrix(jerk)
#     disp = np.array(disp)
#     discard = np.array(discard)
#     start_indices = np.array(start_indices)
#     end_indices = np.array(end_indices)
#     # p2p_id = np.array(p2p_id)
    
#     # discard mvmt elems
#     subjs = subjs[~discard]
#     locs = locs[~discard]
#     axes = axes[~discard]
#     aff_sides = aff_sides[~discard]
#     dur = dur[~discard]
#     vel = vel[~discard]
#     norm_vel = norm_vel[~discard]
#     acc = acc[~discard]
#     jerk = jerk[~discard]
#     disp = disp[~discard]
#     start_indices = start_indices[~discard]
#     end_indices = end_indices[~discard]
#     # p2p_id = p2p_id[~discard]
    
#     # overlap
#     exclusion_mask = overlap(subjs, locs, axes, start_indices, end_indices)
#     subjs = subjs[~exclusion_mask]
#     locs = locs[~exclusion_mask]
#     axes = axes[~exclusion_mask]
#     aff_sides = aff_sides[~exclusion_mask]
#     dur = dur[~exclusion_mask]
#     vel = vel[~exclusion_mask]
#     norm_vel = norm_vel[~exclusion_mask]
#     acc = acc[~exclusion_mask]
#     jerk = jerk[~exclusion_mask]
#     disp = disp[~exclusion_mask]
#     start_indices = start_indices[~exclusion_mask]
#     end_indices = end_indices[~exclusion_mask]
    
#     output = {
#         'subjs': subjs,
#         'locs': locs,
#         'axes': axes,
#         'aff_sides': aff_sides,
#         'dur': dur,
#         'vel': vel,
#         'norm_vel': norm_vel,
#         'acc': acc,
#         'jerk': jerk,
#         'disp': disp,
#         'start_indices': start_indices,
#         'end_indices': end_indices
#     }
#     return output


# def include_overlap(subjs, locs, axes, aff_sides, dur, vel, norm_vel, acc, jerk, disp, start_indices, end_indices, annotations):
#     inclusion_mask = np.zeros_like(subjs, dtype=bool)
    
#     for s, l in sorted(set(zip(subjs, locs))):
#         mask = (subjs == s)
#         mask &= (locs == l)
        
#         start_idx = start_indices[mask]
#         end_idx = end_indices[mask]
        
#         # extract annotations
#         annot = annotations[s].copy()
#         annot_mask = annot['Tier'] == l
#         annot = annot[annot_mask]
#         del annot_mask
        
#         # calculate overlap
#         results = []
#         for start, end in zip(start_idx, end_idx):
#             overlap = (np.minimum(annot['End'], end) - np.maximum(annot['Begin'], start)) / (end - start)
#             if np.any(overlap == 1):
#                 results.append(True)
#             else:
#                 results.append(False)
        
#         # save
#         inclusion_mask[mask] = results
    
#     # further analysis
#     subjs = subjs[inclusion_mask]
#     locs = locs[inclusion_mask]
#     axes = axes[inclusion_mask]
#     aff_sides = aff_sides[inclusion_mask]
#     dur = dur[inclusion_mask]
#     vel = vel[inclusion_mask]
#     norm_vel = norm_vel[inclusion_mask]
#     acc = acc[inclusion_mask]
#     jerk = jerk[inclusion_mask]
#     disp = disp[inclusion_mask]
#     start_indices = start_indices[inclusion_mask]
#     end_indices = end_indices[inclusion_mask]
#     return subjs, locs, axes, aff_sides, dur, vel, norm_vel, acc, jerk, disp, start_indices, end_indices
                       