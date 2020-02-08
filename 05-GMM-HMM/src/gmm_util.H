
//  $Id: gmm_util_simple.H,v 1.1 2009/10/02 01:56:09 stanchen Exp $

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @file gmm_util.H
 *   @brief Miscellaneous routines for manipulating GmmSet's.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _GMM_UTIL_H
#define _GMM_UTIL_H

#include "util.H"

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   GMM statistics for reestimating a GmmSet.
 *
 *   This object stores and updates the statistics required for reestimating
 *   a set of GMM's.  The method #reestimate() reestimates the
 *   associated GmmSet.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class GmmStats {
 public:
  /** Ctor; @p gmmSet is the object to reestimate and @p params
   *   holds parameters controlling how reestimation is done.
   **/
  GmmStats(GmmSet& gmmSet, const map<string, string>& params = ParamsType());

  /** Resets all counts to 0. **/
  void clear();

  /** Updates stats for the @p gmmIdx-th GMM.
   *   The value @p posterior should be the posterior prob/count
   *   of this GMM, and @p feats should be the acoustic
   *   feature vector for the current frame.
   *   Returns log prob base e of GMM for that feature vector.
   *   GMM's are numbered starting from 0.
   **/
  double add_gmm_count(unsigned gmmIdx, double posterior,
                       const vector<double>& feats);

  /** Update stats for all GMM counts in @p gmmCountList given
   *   that @p feats holds the feature vectors for the whole utterance.
   *   Returns the sum of the log probs base e of each GMM in the
   *   list for its corresponding frame; this value will only be
   *   meaningful in a Viterbi-style update.
   **/
  double update(const vector<GmmCount>& gmmCountList,
                const matrix<double>& feats);

  /** Reestimates GmmSet provided in ctor using current stats. **/
  void reestimate() const;

 private:
  /** For storing program parameters that were passed in. **/
  map<string, string> m_params;

  /** Reference to associated GmmSet. **/
  GmmSet& m_gmmSet;

  /** Total counts of each Gaussian. **/
  vector<double> m_gaussCounts;

  /** First-order stats for each dim of each Gaussian. **/
  matrix<double> m_gaussStats1;

  /** Second-order stats for each dim of each Gaussian. **/
  matrix<double> m_gaussStats2;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#endif
