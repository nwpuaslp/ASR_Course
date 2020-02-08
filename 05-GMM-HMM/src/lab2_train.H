
//  $Id: lab2_train.H,v 1.2 2009/10/02 00:31:58 stanchen Exp $

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @file lab2_train.H
 *   @brief Main loop for Lab 2 Viterbi trainer.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _LAB2_TRAIN_H
#define _LAB2_TRAIN_H

#include "front_end.H"
#include "util.H"

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Encapsulation of main loop for Viterbi training.
 *
 *   Holds global variables and has routines for initializing variables
 *   and updating them for each utterance.
 *   We do this so that we can call this code from Java as well.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class Lab2TrainMain {
 public:
  /** Initialize all data given parameters. **/
  Lab2TrainMain(const map<string, string>& params);

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

  /** Returns acoustic model. **/
  GmmSet& get_gmm_set() { return m_gmmSet; }

  /** Returns list of GMM counts. **/
  const vector<GmmCount>& get_gmm_counts() const { return m_gmmCountList; }

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

  /** Input audio for current utterance. **/
  matrix<double> m_inAudio;

  /** Feature vectors for current utterance. **/
  matrix<double> m_feats;

  /** Alignment for current utterance. **/
  vector<int> m_gmmList;

  /** GMM counts for current utterance. **/
  vector<GmmCount> m_gmmCountList;

  /** Total number of iterations. **/
  int m_iterCnt;

  /** Current iteration. **/
  int m_iterIdx;

  /** Total frames processed so far. **/
  int m_totFrmCnt;

  /** Total log prob of utterances processed so far. **/
  double m_totLogProb;

  /** Stream for reading audio data. **/
  ifstream m_audioStrm;

  /** Stream for reading alignments. **/
  ifstream m_alignStrm;

  /** ID string for current utterance. **/
  string m_idStr;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#endif
