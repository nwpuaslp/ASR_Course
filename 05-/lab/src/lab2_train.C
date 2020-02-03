
//  $Id: lab2_train.C,v 1.16 2009/10/02 00:31:58 stanchen Exp $

#include "front_end.H"
#include "gmm_util.H"
#include "lab2_train.H"
#include "util.H"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

Lab2TrainMain::Lab2TrainMain(const map<string, string>& params)
    : m_params(params),
      m_frontEnd(m_params),
      m_gmmSet(get_required_string_param(m_params, "in_gmm")),
      m_outGmmFile(get_required_string_param(m_params, "out_gmm")),
      m_iterCnt(get_int_param(m_params, "iters", 1)),
      m_iterIdx(1),
      m_totFrmCnt(0),
      m_totLogProb(0.0) {}

bool Lab2TrainMain::init_iter() {
  if (m_iterIdx > m_iterCnt) return false;
  m_audioStrm.clear();
  m_audioStrm.open(get_required_string_param(m_params, "audio_file").c_str());
  m_alignStrm.clear();
  m_alignStrm.open(get_required_string_param(m_params, "align_file").c_str());
  m_totFrmCnt = 0;
  m_totLogProb = 0.0;
  return true;
}

void Lab2TrainMain::finish_iter() {
  m_audioStrm.close();
  m_alignStrm.close();
  cout << format("Iteration %d: %.6f logprob/frame (%d frames)") % m_iterIdx %
              (m_totFrmCnt ? m_totLogProb / m_totFrmCnt : 0.0) % m_totFrmCnt
       << endl;
  ++m_iterIdx;
}

bool Lab2TrainMain::init_utt() {
  if (m_audioStrm.peek() == EOF) return false;
  m_idStr = read_float_matrix(m_audioStrm, m_inAudio);
  m_frontEnd.get_feats(m_inAudio, m_feats);
  if (m_feats.size2() != m_gmmSet.get_dim_count())
    throw runtime_error("Mismatch in GMM and feat dim.");
  if (m_alignStrm.peek() == EOF)
    throw runtime_error(
        "Mismatch in number of audio files "
        "and alignments.");
  read_int_vector(m_alignStrm, m_gmmList, m_idStr);
  if (m_gmmList.size() != m_feats.size1())
    throw runtime_error("Mismatch in alignment and feat lengths.");
  int gmmCnt = m_gmmSet.get_gmm_count();
  int frmCnt = m_gmmList.size();
  m_gmmCountList.clear();
  for (int frmIdx = 0; frmIdx < frmCnt; ++frmIdx) {
    int gmmIdx = m_gmmList[frmIdx];
    if ((gmmIdx < 0) || (gmmIdx >= gmmCnt))
      throw runtime_error("Out of range GMM index.");
    // posterior is 1.0 in Viterbi Training and 1-component GMM
    m_gmmCountList.push_back(GmmCount(gmmIdx, frmIdx, 1.0));
  }
  return true;
}

void Lab2TrainMain::finish_utt(double logProb) {
  m_totFrmCnt += m_feats.size1();
  m_totLogProb += logProb;
}

void Lab2TrainMain::finish() { m_gmmSet.write(m_outGmmFile); }

#ifndef NO_MAIN_LOOP

void main_loop(const char** argv) {
  map<string, string> params;
  process_cmd_line(argv, params);

  Lab2TrainMain mainObj(params);
  GmmStats gmmStats(mainObj.get_gmm_set(), params);
  while (mainObj.init_iter()) {
    gmmStats.clear();
    while (mainObj.init_utt()) {
      double logProb =
          gmmStats.update(mainObj.get_gmm_counts(), mainObj.get_feats());
      mainObj.finish_utt(logProb);
    }
    mainObj.finish_iter();
    gmmStats.reestimate();
  }
  mainObj.finish();
}

#endif

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
