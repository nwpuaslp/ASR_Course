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
    assert (len(feat) == len(text))
    dict_utt2feat = {}
    dict_utt2target = {}
    for i in range(len(feat)):
        utt_id1, ark = feat[i].strip('\n').split(' ')
        utt_id2, target = text[i].strip('\n').split(' ')
        assert (utt_id1 == utt_id2)
        dict_utt2feat[utt_id1] = ark
        dict_utt2target[utt_id2] = target
    return dict_utt2feat, dict_utt2target


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


def cmvn(mat):
    mean = np.mean(mat, axis=0)
    var = np.var(mat, axis=0)
    mat = (mat - mean) / var
    return mat


def splice(feats, left_context, right_context):
    ''' Splice feature
    Args:
        feats: input feats
        left_context: left context for splice
        right_context: right context for splice
    Returns:
        Spliced feature
    '''
    if left_context == 0 and right_context == 0:
        return feats
    assert (len(feats.shape) == 2)
    num_rows = feats.shape[0]
    first_frame = feats[0]
    last_frame = feats[-1]
    padding_feats = feats
    if left_context > 0:
        left_padding = np.vstack([first_frame for i in range(left_context)])
        padding_feats = np.vstack((left_padding, padding_feats))
    if right_context > 0:
        right_padding = np.vstack([last_frame for i in range(right_context)])
        padding_feats = np.vstack((padding_feats, right_padding))
    outputs = []
    for i in range(num_rows):
        splice_feats = np.hstack([
            padding_feats[i]
            for i in range(i, i + 1 + left_context + right_context)
        ])
        outputs.append(splice_feats)
    return np.vstack(outputs)


def build_input(targets_mapping, utt2feat, utt2target):
    mat_list = []
    label_list = []
    for utt in utt2feat:
        assert (utt in utt2target)
        ark = utt2feat[utt]
        t = utt2target[utt]
        mat = kaldi_io.read_mat(ark)
        mat = splice(mat, 5, 5)
        mat_list.append(mat)
        label_list.extend([targets_mapping[t]] * mat.shape[0])
    return np.concatenate(mat_list, axis=0), np.array(label_list)
