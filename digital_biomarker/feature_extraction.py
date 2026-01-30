#################################
# extract features using
# each movement
# segmentation method
#################################
import os
import gc
import copy
import argparse
import itertools
from scipy.stats import pearsonr, spearmanr

import warnings
warnings.filterwarnings("ignore")

from utils import *
import scripts.kinematic_approach as kinematic_approach
import scripts.features as mvmt_elem_features
# from preprocess import window

work_dir = os.path.join(
    '/work',
    'pi_sunghoonlee_umass_edu',
    'Yunda'
)

# input_dir has stroke data 
# with generated orientations
input_dir = os.path.join(
    work_dir,
    'kinematic_measures'
)

# output_dir = os.path.join(
#     work_dir,
#     'kinematic_measures'
# )
output_dir = 'output'


def main():
    # Load parameters
    args = parse_arguments()
    print(f'AHRS_FILTER_TYPE:       {args.ahrs_filter_type}', flush=True)
    print(f'RFUNC:                  {args.rfunc}', flush=True)
    print(f'TIMESTAMP:              {args.timestamp}', flush=True)
    print(f'ICC_TYPE:               {args.icc_type}', flush=True)
    print(f'ICC_THRESH:             {args.icc_thresh}', flush=True)
    print(f'INPUT_WND_DATA:         {args.input_wnd_data}', flush=True)
    ICC_THRESH = args.icc_thresh
    
    # Convert args.rfunc to functions
    if args.rfunc == 'pearsonr':
        args.rfunc = pearsonr
    elif args.rfunc == 'spearmanr':
        args.rfunc = spearmanr
        
    # # Check if the npz file is in input_dir
    # if 'stroke_data_%s.npz' % args.ahrs_filter_type not in os.listdir(input_dir):
    #     # Load stroke_data_copy
    #     input_filename = 'stroke_data.npz'
    #     data = np.load(os.path.join(input_dir, input_filename), allow_pickle=True)
    #     stroke_data_copy = data['data'].item()
    #     del data
    #     
    #     print('Generating orient.', flush=True)
    #     orient_estimation(stroke_data_copy, args.ahrs_filter_type, output_dir)
    
    # Load annotations
    annotation_file = os.path.join('input', 'stroke_annotationUPDATE.npz')
    data = np.load(annotation_file, allow_pickle=True)['arr_0']
    annotations = data.item()
    del annotation_file
    
    # Load scores
    subject_list = [1, 2, 3, 4, 5, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20]
    score_keys = ['FM', 'MAL_AoU', 'MAL_QoM', 'WMFT_TR', 'WMFT_FAS']
    score_dict = dict()
    for score_key in score_keys:
        score_dict[score_key] = np.array([getattr(demographics[s], score_key) for s in subject_list])
        del score_key

    # Prepare for training
    feature_dict = {}
    
    # Load data
    data = np.load(os.path.join(input_dir, 'stroke_data_%s.npz' % args.ahrs_filter_type), allow_pickle=True)
    stroke_data = data['data'].item()
    del data
    
        
    #####################################################
    # Method: extract features using linear
    # segmentation
    #####################################################
    # load stroke windows
    windows = load_stroke_windows(os.path.join(
        'input',
        args.input_wnd_data))
    
    # load predictions
    filepath = os.path.join('input', 'predictions2_{:s}'.format(args.input_wnd_data))
    nn_results = summarize_output(filepath, subject_list)
    
    #####################################################
    # Process stroke data
    # load stroke data
    stroke_data_copy = copy.deepcopy(stroke_data)

    # rotation
    num_devices = 1
    stroke_data_copy = rotation(stroke_data_copy, num_devices=num_devices, ahrs_filter_type=args.ahrs_filter_type)

    # trim data
    stroke_data_copy = trim_data(stroke_data_copy, annotations, sensor_offset)

    # filtering data
    stroke_data_copy = filtering(stroke_data_copy, filter_type='lowpass')
    #####################################################
    
    kwargs = {'n_components': 1}
    seg_method_attempts = ['p2p_true', 'p2p_pred']
    p_score_attempts = [-1] # [-1, 0.7, 0.75, 0.8, 0.85]
    delta_attempts = np.arange(0, 1+.001, 0.1)
    for seg_method, p_score, delta in itertools.product(seg_method_attempts, p_score_attempts, delta_attempts):
        if seg_method == 'p2p_true' and p_score != -1:
            continue

        delta = np.around(delta, 2)
                
        print('seg. method: {:s}, num_devices: {:d}'.format(seg_method, num_devices), flush=True)
        print('p_score: {:.2f}, delta: {:.2f}'.format(p_score, delta), flush=True)
    
        # tuning op
        nn_results_copy = tuning_op(nn_results, p_score, seg_method=seg_method)
        
        # organize predictions
        wnd_true, wnd_pred = organize_predictions(windows, nn_results_copy)
        
        # post-processing
        post_pred = combine_windows(windows, wnd_pred)
        
        # convert post_pred into annotation format
        pred_annotations = convert_pred_to_annotation(post_pred)
        
        # segmentation
        results = segmentation(stroke_data_copy, pred_annotations, seg_method, delta, args=args, **kwargs)

        # feature extraction
        subjs, locs, axes, aff_sides, dur, vel, norm_vel, acc, jerk, disp, start_indices, end_indices, p2p_id = organize_mvmt_elems(results)
        elements = [subjs, locs, axes, aff_sides, dur, vel, norm_vel, acc, jerk, disp, start_indices, end_indices]
        
        # Extract mvmt elems from affected limbs
        subjs = subjs[aff_sides]
        locs = locs[aff_sides]
        axes = axes[aff_sides]
        dur = dur[aff_sides]
        vel = vel[aff_sides]
        norm_vel = norm_vel[aff_sides]
        acc = acc[aff_sides]
        jerk = jerk[aff_sides]
        disp = disp[aff_sides]
        start_indices = start_indices[aff_sides]
        end_indices = end_indices[aff_sides]
        p2p_id = p2p_id[aff_sides]
        
        # Extract features
        homog = None
        feature_names, subjects, features = mvmt_elem_features.extract_features(
            subjs, homog, vel, norm_vel, acc, jerk, dur, disp, annotations, 'AFF')
        
        # Remove invalid features
        features, feature_names = remove_invalid_features(features, feature_names)
    
        # Reliability Test
        indices_1st, indices_2nd = split_data(subjs, locs, axes, split_method=1.5, stroke_data=stroke_data_copy, start_indices=start_indices, p2p_id=p2p_id)
        
        # Extract features
        feature_names_1st, subjects_1st, features_1st = mvmt_elem_features.extract_features(
            subjs[indices_1st], homog, vel[indices_1st], norm_vel[indices_1st], acc[indices_1st], 
            jerk[indices_1st], dur[indices_1st], disp[indices_1st], annotations, 'AFF')

        feature_names_2nd, subjects_2nd, features_2nd = mvmt_elem_features.extract_features(
            subjs[indices_2nd], homog, vel[indices_2nd], norm_vel[indices_2nd], acc[indices_2nd], 
            jerk[indices_2nd], dur[indices_2nd], disp[indices_2nd], annotations, 'AFF')
        del indices_1st, indices_2nd

        # Remove invalide features
        features_1st, feature_names_1st = remove_invalid_features(features_1st, feature_names_1st)
        features_2nd, feature_names_2nd = remove_invalid_features(features_2nd, feature_names_2nd)
        
        if features_1st.shape[0] != features_2nd.shape[0]:
            continue
            
        # Reliability test
        dataframe = reliability_test(features, features_1st, features_2nd, feature_names, 
                                     feature_names_1st, feature_names_2nd, score_dict, args)
        
        sorted_dataframe = dataframe[dataframe['icc'] >= ICC_THRESH].copy()
        sorted_dataframe = sorted_dataframe.sort_values(by=['corr'], key=lambda x:np.abs(x), ascending=False)
        # print(sorted_dataframe.iloc[:5], flush=True)
        del sorted_dataframe
        
        # Save
        key = (args.ahrs_filter_type, seg_method)
        if key not in feature_dict:
            feature_dict[key] = {
                'features': features,
                'feature_names': feature_names,
                'subjects': subjects,
                'features_1st': features_1st,
                'feature_names_1st': feature_names_1st,
                'features_2nd': features_2nd,
                'feature_names_2nd': feature_names_2nd,
                'dataframe': dataframe,
                'best_delta': delta,
                'elements': elements,
                'precision_score': p_score
            }
        else:
            # compare the current best feature with the saved best feature
            curr_df = dataframe.sort_values(by=['corr'], key=lambda x:np.abs(x), ascending=False)
            curr_df = curr_df[curr_df['icc'] >= ICC_THRESH]
            curr_corr = curr_df['corr'].iloc[0] if len(curr_df) else 0.0
            print('curr_corr: {:.2f}'.format(curr_corr), flush=True)
            
            prev_df = feature_dict[key]['dataframe'].sort_values(by=['corr'], key=lambda x:np.abs(x), ascending=False)
            prev_df = prev_df[prev_df['icc'] >= ICC_THRESH]
            prev_corr = prev_df['corr'].iloc[0] if len(prev_df) else 0.0
            print('prev_corr: {:.2f}'.format(prev_corr), flush=True)
            
            if np.abs(curr_corr) > np.abs(prev_corr):
                feature_dict[key] = {
                    'features': features,
                    'feature_names': feature_names,
                    'subjects': subjects,
                    'features_1st': features_1st,
                    'feature_names_1st': feature_names_1st,
                    'features_2nd': features_2nd,
                    'feature_names_2nd': feature_names_2nd,
                    'dataframe': dataframe,
                    'best_delta': delta,
                    'elements': elements,
                    'precision_score': p_score
                }
            del curr_df, curr_corr, prev_df, prev_corr

        # del stroke_data_copy
        del results
        del features, feature_names
        del features_1st, feature_names_1st
        del features_2nd, feature_names_2nd
        del dataframe
        gc.collect()
        print('\n', flush=True) 

    # Save
    output_folder = os.path.join(
        output_dir,
        '{:s}_{:s}'.format(args.rfunc.__name__, args.corr_target))
    os.makedirs(output_folder, exist_ok=True)
    output_name = os.path.join(
        output_folder,
        'feature_dict_{:s}_{:s}_ICC_{:s}_{:.2f}_{:s}_split_method_1_1.5.npz'.format(
            args.timestamp,
            args.ahrs_filter_type,
            args.icc_type,
            args.icc_thresh,
            args.input_wnd_data
        ))
    
    np.savez(output_name, **{'data': feature_dict})
    print('DONE.', flush=True)
    
    
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ahrs_filter_type',
        required=True,
        type=str,
        choices=['angularrate', 'aqua', 'complementary', 'madgwick', 'mahony']
    )
    parser.add_argument(
        '--rfunc',
        required=True,
        type=str,
        choices=['pearsonr', 'spearmanr']
    )
    parser.add_argument(
        '--timestamp',
        required=True,
        type=str
    )
    parser.add_argument(
        '--corr_target',
        required=True,
        type=str,
        default='FM',
        choices=['FM', 'objective', 'all']
    )
    parser.add_argument(
        '--icc_type',
        required=False,
        type=str,
        default='(2, 1)',
        choices=['(3, 1)', '(2, 1)']
    )
    parser.add_argument(
        '--icc_thresh',
        required=False,
        type=float,
        default=0.75,
    )
    parser.add_argument(
        '--input_wnd_data',
        choices=['wnd_sz_2_overlap_1_4', 'wnd_sz_3_overlap_1_3']
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()