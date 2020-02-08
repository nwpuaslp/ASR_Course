
//  $Id: gmm_util_simple.C,v 1.3 2009/10/02 04:06:44 stanchen Exp $

#include "gmm_util.H"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

GmmStats::GmmStats(GmmSet& gmmSet, const map<string, string>& params)
    : m_params(params),
      m_gmmSet(gmmSet),
      m_gaussCounts(m_gmmSet.get_gaussian_count()),
      m_gaussStats1(m_gmmSet.get_gaussian_count(), m_gmmSet.get_dim_count()),
      m_gaussStats2(m_gmmSet.get_gaussian_count(), m_gmmSet.get_dim_count()) {
  clear();
}

void GmmStats::clear() {
  fill(m_gaussCounts.begin(), m_gaussCounts.end(), 0.0);
  fill(m_gaussStats1.data().begin(), m_gaussStats1.data().end(), 0.0);
  fill(m_gaussStats2.data().begin(), m_gaussStats2.data().end(), 0.0);
}

double GmmStats::add_gmm_count(unsigned gmmIdx, double posterior,
                               const vector<double>& feats) {
  if (m_gmmSet.get_component_count(gmmIdx) != 1)
    throw runtime_error("GMM doesn't have single component.");
  int gaussIdx = m_gmmSet.get_gaussian_index(gmmIdx, 0);
  int dimCnt = m_gmmSet.get_dim_count();

  //  BEGIN_LAB
  //
  //  Input:
  //      "dimCnt" holds the dimension of the Gaussian and the
  //      acoustic feature vector.
  //      The acoustic feature vector is held in
  //      "feats[0 .. (dimCnt-1)]".
  //      "gaussIdx" is the index of the Gaussian to be updated.
  //      "posterior" is the posterior count of this Gaussian for
  //      the current frame.
  //
  //      The values of the current means and variances can be
  //      accessed via the object "m_gmmSet".
  //
  //  Output:
  //      You should update the counts stored in
  //
  //      m_gaussCounts[0 .. (#gaussians-1)]
  //      m_gaussStats1(0 .. (#gaussians-1), 0 .. (dimCnt - 1))
  //      m_gaussStats2(0 .. (#gaussians-1), 0 .. (dimCnt - 1))
  //
  //      "m_gaussCounts" is intended to hold the total occupancy count
  //      of each Gaussian; "m_gaussStats1" is intended for
  //      storing some sort of first-order statistic for each
  //      dimension of each Gaussian; and "m_gaussStats2" is intended for
  //      storing some sort of second-order statistic for each
  //      dimension of each Gaussian.  The statistics you take
  //      need to be sufficient for doing the reestimation step below.
  //
  //      These counts have all been initialized to zero
  //      somewhere else at the appropriate time.

  // suppose each GMM only has one component
  //  END_LAB
  //

  return 0.0;
}

double GmmStats::update(const vector<GmmCount>& gmmCountList,
                        const matrix<double>& feats) {
  unsigned frmCnt = feats.size1();
  unsigned gmmCnt = m_gmmSet.get_gmm_count();
  unsigned lastFrmIdx = (unsigned)-1;
  vector<double> frameBuf;
  double logProb = 0.0;
  for (unsigned cntIdx = 0; cntIdx < gmmCountList.size(); ++cntIdx) {
    const GmmCount& gmmCount = gmmCountList[cntIdx];
    unsigned curFrmIdx = gmmCount.get_frame_index();
    unsigned gmmIdx = gmmCount.get_gmm_index();
    if ((curFrmIdx >= frmCnt) || (gmmIdx >= gmmCnt))
      throw runtime_error(
          "Out-of-bounds frame index or GMM index "
          "in GMM count.");
    if (curFrmIdx != lastFrmIdx)
      copy_matrix_row_to_vector(feats, curFrmIdx, frameBuf);
    logProb += add_gmm_count(gmmIdx, gmmCount.get_count(), frameBuf);
    lastFrmIdx = curFrmIdx;
  }
  return logProb;
}

void GmmStats::reestimate() const {
  //  Reestimate Gaussian means and variances.
  int gaussCnt = m_gmmSet.get_gaussian_count();
  int dimCnt = m_gmmSet.get_dim_count();

  //  BEGIN_LAB
  //
  //  Input:
  //      "gaussCnt" holds the total number of Gaussians.
  //      "dimCnt" holds the dimension of the Gaussians.
  //
  //      The counts you have collected above are stored in:
  //
  //      m_gaussCounts[0 .. (#gaussians-1)]
  //      m_gaussStats1(0 .. (#gaussians-1), 0 .. (dimCnt - 1))
  //      m_gaussStats2(0 .. (#gaussians-1), 0 .. (dimCnt - 1))
  //
  //  Output:
  //      You should call the functions:
  //
  //      m_gmmSet.set_gaussian_mean(gaussIdx, dimIdx, newMean);
  //      m_gmmSet.set_gaussian_var(gaussIdx, dimIdx, newVar);
  //
  //      for each dimension of each Gaussian with the reestimated
  //      values of the means and variances.

  //  END_LAB
  //
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void init_simple_gmms(GmmSet& gmmSet, unsigned gmmCnt, unsigned dimCnt) {
  vector<int> gmmCompCounts(gmmCnt, 1);
  gmmSet.init(gmmCompCounts, dimCnt);
}

void split_gmms(const GmmSet& srcGmmSet, GmmSet& dstGmmSet,
                const map<string, string>& params) {
  double perturbFactor = get_float_param(params, "perturb", 0.2);
  double minPerturb = get_float_param(params, "min_perturb", 0.001);
  int gmmCnt = srcGmmSet.get_gmm_count();
  int dimCnt = srcGmmSet.get_dim_count();
  int srcGaussCnt = srcGmmSet.get_gaussian_count();
  vector<int> gmmCompCounts, compMap;
  for (int gmmIdx = 0; gmmIdx < gmmCnt; ++gmmIdx) {
    int compCnt = srcGmmSet.get_component_count(gmmIdx);
    gmmCompCounts.push_back(compCnt * 2);
    for (int compIdx = 0; compIdx < compCnt; ++compIdx) {
      int srcGaussIdx = srcGmmSet.get_gaussian_index(gmmIdx, compIdx);
      compMap.push_back(2 * srcGaussIdx);
      compMap.push_back(2 * srcGaussIdx + 1);
    }
  }
  dstGmmSet.init(gmmCompCounts, dimCnt, compMap);
  assert((int)dstGmmSet.get_gaussian_count() == srcGaussCnt * 2);
  for (int gmmIdx = 0; gmmIdx < gmmCnt; ++gmmIdx) {
    int compCnt = srcGmmSet.get_component_count(gmmIdx);
    for (int compIdx = 0; compIdx < compCnt; ++compIdx) {
      double wgt = srcGmmSet.get_component_weight(gmmIdx, compIdx);
      dstGmmSet.set_component_weight(gmmIdx, 2 * compIdx, wgt / 2.0);
      dstGmmSet.set_component_weight(gmmIdx, 2 * compIdx + 1, wgt / 2.0);
    }
  }
  for (int srcGaussIdx = 0; srcGaussIdx < srcGaussCnt; ++srcGaussIdx) {
    int dstGaussIdx = srcGaussIdx * 2;
    dstGmmSet.copy_gaussian(dstGaussIdx, srcGmmSet, srcGaussIdx);
    dstGmmSet.copy_gaussian(dstGaussIdx + 1, srcGmmSet, srcGaussIdx);
    for (int dimIdx = 0; dimIdx < dimCnt; ++dimIdx) {
      double curMean = srcGmmSet.get_gaussian_mean(srcGaussIdx, dimIdx);
      double curVar = srcGmmSet.get_gaussian_var(srcGaussIdx, dimIdx);
      double perturbMean = perturbFactor * sqrt((curVar > 0.0) ? curVar : 0.0);
      if (perturbMean < minPerturb) perturbMean = minPerturb;
      dstGmmSet.set_gaussian_mean(dstGaussIdx, dimIdx, curMean - perturbMean);
      dstGmmSet.set_gaussian_mean(dstGaussIdx + 1, dimIdx,
                                  curMean + perturbMean);
    }
  }
}

void expand_gmms_ci_to_cd(const GmmSet& srcGmmSet, GmmSet& dstGmmSet,
                          const vector<int>& phoneGmmCounts) {
  int srcGmmCnt = srcGmmSet.get_gmm_count();
  if ((int)phoneGmmCounts.size() != srcGmmCnt)
    throw runtime_error("Invalid GMM count array.");
  int dimCnt = srcGmmSet.get_dim_count();
  vector<int> gmmCompCounts;
  for (int srcGmmIdx = 0; srcGmmIdx < srcGmmCnt; ++srcGmmIdx) {
    int phoneGmmCnt = phoneGmmCounts[srcGmmIdx];
    if (phoneGmmCnt < 1)
      throw runtime_error("Nonpositive num GMM's for a phone.");
    gmmCompCounts.insert(gmmCompCounts.end(), phoneGmmCnt,
                         srcGmmSet.get_component_count(srcGmmIdx));
  }
  dstGmmSet.init(gmmCompCounts, dimCnt);

  int dstGmmIdx = 0;
  for (int srcGmmIdx = 0; srcGmmIdx < srcGmmCnt; ++srcGmmIdx) {
    int compCnt = srcGmmSet.get_component_count(srcGmmIdx);
    int phoneGmmCnt = phoneGmmCounts[srcGmmIdx];
    for (int phoneGmmIdx = 0; phoneGmmIdx < phoneGmmCnt; ++phoneGmmIdx) {
      for (int compIdx = 0; compIdx < compCnt; ++compIdx) {
        dstGmmSet.set_component_weight(
            dstGmmIdx, compIdx,
            srcGmmSet.get_component_weight(srcGmmIdx, compIdx));
        dstGmmSet.copy_gaussian(
            dstGmmSet.get_gaussian_index(dstGmmIdx, compIdx), srcGmmSet,
            srcGmmSet.get_gaussian_index(srcGmmIdx, compIdx));
      }
      ++dstGmmIdx;
    }
  }
  assert(dstGmmIdx == (int)dstGmmSet.get_gmm_count());
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
