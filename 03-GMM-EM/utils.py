import kaldi_io
import numpy as np

def read_all_data(feat_scp):
    feat_fid = open(feat_scp, 'r')
    feat = feat_fid.readlines()
    feat_fid.close()
    mat_list = []

    for i in range(len(feat)):
        _, ark = feat[i].split()
        mat = kaldi_io.read_mat(ark)
        mat_list.append(mat)
    return np.concatenate(mat_list, axis=0)

def read_feats_and_targets(feat_scp, text_file):
    feat_fid = open(feat_scp, 'r')
    text_fid = open(text_file, 'r')
    feat = feat_fid.readlines()
    text = text_fid.readlines()
    feat_fid.close()
    text_fid.close()
    assert(len(feat) == len(text))
    dict_utt2feat = {}
    dict_target2utt = {}
    for i in range(len(feat)):
        utt_id1, ark = feat[i].strip('\n').split(' ')
        utt_id2, target = text[i].strip('\n').split(' ')
        dict_utt2feat[utt_id1] = ark
        if target in dict_target2utt.keys():
            dict_target2utt[target].append(utt_id2)
        else:
            dict_target2utt[target] = [utt_id2]
    return dict_utt2feat, dict_target2utt

def get_feats(target, dic_utt2feat, dict_target2utt):
    """ Read feats for a specific target
        :param target: char, '0', '1', ..., '9', o'
        :param dict_utt2feat: utterance to feat dictionary
        :param dict_target2utt: target to utterance dictionary
        :return: feature matrix for this target, num_samples * feature dim
    """
    mat_list = []
    for utt in dict_target2utt[target]:
        ark = dic_utt2feat[utt]
        mat = kaldi_io.read_mat(ark)
        mat_list.append(mat)
    return np.concatenate(mat_list, axis=0)

