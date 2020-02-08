
//  $Id: lab2_fb.H,v 1.3 2009/10/03 05:26:15 stanchen Exp stanchen $

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @file lab2_fb.H
 *   @brief Main loop for Lab 2 Forward-Backward trainer.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _LAB2_FB_H
#define _LAB2_FB_H

#include "front_end.H"
#include "util.H"

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Cell in dynamic programming chart for Forward-Backward algorithm.
 *
 *   Holds forward and backward log probs.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class FbCell {
 public:
  /** Ctor; inits F+B log probs to g_zeroLogProb. **/
  FbCell() : m_forwLogProb(g_zeroLogProb), m_backLogProb(g_zeroLogProb) {}

#ifndef SWIG
#ifndef DOXYGEN
  //  Hack; for bug in matrix<> class in boost 1.32.
  explicit FbCell(int)
      : m_forwLogProb(g_zeroLogProb), m_backLogProb(g_zeroLogProb) {}
#endif
#endif

  /** Sets forward log prob of cell. **/
  void set_forw_log_prob(double logProb) { m_forwLogProb = logProb; }

  /** Sets backward log prob of cell. **/
  void set_back_log_prob(double logProb) { m_backLogProb = logProb; }

  /** Returns forward log prob of cell. **/
  double get_forw_log_prob() const { return m_forwLogProb; }

  /** Returns backward log prob of cell. **/
  double get_back_log_prob() const { return m_backLogProb; }

 private:
  /** Forward logprob. **/
  double m_forwLogProb;

  /** Backward logprob. **/
  double m_backLogProb;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/** Initializes chart for backward pass. **/
double init_backward_pass(const Graph& graph, matrix<FbCell>& chart);

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Encapsulation of main loop for Forward-Backward decoding.
 *
 *   Holds global variables and has routines for initializing variables
 *   and updating them for each utterance.
 *   We do this so that we can call this code from Java as well.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class Lab2FbMain {
 public:
  /** Initialize all data given parameters. **/
  Lab2FbMain(const map<string, string>& params);

  /** Called at the beginning of processing each iteration.
   *   Returns whether at EOF.
   **/
  bool init_iter();

  /** Called at the end of processing each iteration. **/
  void finish_iter();

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
  matrix<FbCell>& get_chart() { return m_chart; }

  /** Returns acoustic model. **/
  GmmSet& get_gmm_set() { return m_gmmSet; }

  /** Returns object for storing GMM counts. **/
  vector<GmmCount>& get_gmm_counts() { return m_gmmCountList; }

  /** Returns object for storing transition counts. **/
  map<int, double>& get_trans_counts() { return m_transCounts; }

  /** Returns feature vectors. **/
  const matrix<double>& get_feats() const { return m_feats; }

 private:
  /** Program parameters. **/
  map<string, string> m_params;

  /** Front end. **/
  FrontEnd m_frontEnd;

  /** Acoustic model. **/
  GmmSet m_gmmSet;

  /** Place to output model. **/
  string m_outGmmFile;

  /** Stream for reading audio data. **/
  ifstream m_audioStrm;

  /** Graph/HMM. **/
  Graph m_graph;

  /** Stream for reading graphs, if doing alignment. **/
  ifstream m_graphStrm;

  /** ID string for current utterance. **/
  string m_idStr;

  /** Input audio for current utterance. **/
  matrix<double> m_inAudio;

  /** Feature vectors for current utterance. **/
  matrix<double> m_feats;

  /** GMM probs for current utterance. **/
  matrix<double> m_gmmProbs;

  /** For storing GMM counts. **/
  vector<GmmCount> m_gmmCountList;

  /** Temporary buffer, For thresholding GMM counts. **/
  vector<GmmCount> m_gmmCountListThresh;

  /** File to output transition counts to, if desired. **/
  string m_transCountsFile;

  /** Transition counts, for training arc probs. **/
  map<int, double> m_transCounts;

  /** DP chart for current utterance. **/
  matrix<FbCell> m_chart;

  /** Total number of iterations. **/
  int m_iterCnt;

  /** Current iteration. **/
  int m_iterIdx;

  /** Total frames processed so far. **/
  int m_totFrmCnt;

  /** Total log prob of utterances processed so far. **/
  double m_totLogProb;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#endif
