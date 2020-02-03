
//  $Id: lab2_vit.H,v 1.4 2009/10/03 05:26:15 stanchen Exp stanchen $

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @file lab2_vit.H
 *   @brief Main loop for Lab 2 Viterbi decoder.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _LAB2_VIT_H
#define _LAB2_VIT_H

#include "front_end.H"
#include "util.H"

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Cell in dynamic programming chart for Viterbi algorithm.
 *
 *   Holds Viterbi log prob; and arc ID of best incoming arc for backtrace.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class VitCell {
 public:
  /** Ctor; inits log prob to g_zeroLogProb and arc ID to -1. **/
  VitCell() : m_logProb(g_zeroLogProb), m_arcId(-1) {}

#ifndef SWIG
#ifndef DOXYGEN
  //  Hack; for bug in matrix<> class in boost 1.32.
  explicit VitCell(int) : m_logProb(g_zeroLogProb), m_arcId(-1) {}
#endif
#endif

  /** Sets associated log prob and arc ID. **/
  void assign(double logProb, int arcId) {
    m_logProb = logProb;
    m_arcId = arcId;
  }

  /** Returns log prob of cell. **/
  double get_log_prob() const { return m_logProb; }

  /** Returns arc ID of cell. **/
  int get_arc_id() const { return m_arcId; }

 private:
  /** Forward Viterbi logprob. **/
  double m_logProb;

  /** ID of best incoming arc, for traceback. **/
  int m_arcId;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/** Routine for Viterbi backtrace. **/
double viterbi_backtrace(const Graph& graph, matrix<VitCell>& chart,
                         vector<int>& outLabelList, bool doAlign);

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Encapsulation of main loop for Viterbi decoding.
 *
 *   Holds global variables and has routines for initializing variables
 *   and updating them for each utterance.
 *   We do this so that we can call this code from Java as well.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class Lab2VitMain {
 public:
  /** Initialize all data given parameters. **/
  Lab2VitMain(const map<string, string>& params);

  /** Called at the beginning of processing each utterance.
   *   Returns whether at EOF.
   **/
  bool init_utt();

  /** Called at the end of processing each utterance. **/
  void finish_utt(double logProb);

  /** Called at end of program. **/
  void finish();

  /** Returns decoding graph/HMM. **/
  const Graph& get_graph() const { return m_graph; }

  /** Returns matrix of GMM log probs for each frame. **/
  const matrix<double>& get_gmm_probs() const { return m_gmmProbs; }

  /** Returns DP chart. **/
  matrix<VitCell>& get_chart() { return m_chart; }

  /** Returns vector to place decoded labels in. **/
  vector<int>& get_label_list() { return m_labelList; }

  /** Returns acoustic weight. **/
  double get_acous_wgt() const { return m_acousWgt; }

  /** Returns whether doing forced alignment or decoding. **/
  bool do_align() const { return m_doAlign; }

 private:
  /** Program parameters. **/
  map<string, string> m_params;

  /** Front end. **/
  FrontEnd m_frontEnd;

  /** Acoustic model. **/
  GmmSet m_gmmSet;

  /** Stream for reading audio data. **/
  ifstream m_audioStrm;

  /** Whether to do forced alignment or decoding. **/
  bool m_doAlign;

  /** Graph/HMM. **/
  Graph m_graph;

  /** Stream for reading graphs, if doing alignment. **/
  ifstream m_graphStrm;

  /** Stream for writing decoding output or alignments. **/
  ofstream m_outStrm;

  /** Acoustic weight. **/
  double m_acousWgt;

  /** ID string for current utterance. **/
  string m_idStr;

  /** Input audio for current utterance. **/
  matrix<double> m_inAudio;

  /** Feature vectors for current utterance. **/
  matrix<double> m_feats;

  /** GMM probs for current utterance. **/
  matrix<double> m_gmmProbs;

  /** DP chart for current utterance. **/
  matrix<VitCell> m_chart;

  /** Decoded output. **/
  vector<int> m_labelList;

  /** Total frames processed so far. **/
  int m_totFrmCnt;

  /** Total log prob of utterances processed so far. **/
  double m_totLogProb;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#endif
