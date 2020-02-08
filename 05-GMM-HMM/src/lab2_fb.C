
//  $Id: lab2_fb.C,v 1.32 2009/10/03 04:47:32 stanchen Exp $

#include "front_end.H"
#include "gmm_util.H"
#include "lab2_fb.H"
#include "util.H"

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Routine for the Forward-Backward algorithm.
 *
 *   @param graph HMM/graph to operate on.
 *   @param gmmProbs Matrix of log prob for each GMM for each frame.
 *   @param chart Dynamic programming chart to fill in; already
 *       allocated to be of correct size and initialized with default values.
 *   @param gmmCountList List of GMM counts to be filled in; this vector
 *       will be empty on entry.
 *   @param transCounts Transition/arc counts to be filled in.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
double forward_backward(const Graph& graph, const matrix<double>& gmmProbs,
                        matrix<FbCell>& chart, vector<GmmCount>& gmmCountList,
                        map<int, double>& transCounts) {
  int frmCnt = chart.size1() - 1;
  int stateCnt = chart.size2();

  //  BEGIN_LAB
  //
  //  Input:
  //      An HMM stored in the object "graph" of type "Graph".
  //      A matrix of doubles "gmmProbs"
  //
  //      gmmProbs(0 .. (frmCnt - 1), 0 .. (#GMM's - 1))
  //
  //      that stores the log prob of each GMM in "gmmSet"
  //      for each frame.
  //
  //  Output:
  //      A matrix "chart" of "FbCell" objects (declaration of
  //      "FbCell" class above):
  //
  //      chart(0 .. frmCnt, 0 .. stateCnt - 1)
  //
  //      On exit, chart(frmIdx, stateIdx).get_forw_log_prob()
  //      should be set to the forward probability
  //      of being in state "stateIdx" given the
  //      first "frmIdx" frames of observations.
  //      If a cell is unreachable from the start state,
  //      this value can be set to "g_zeroLogProb", which is
  //      what these values are initialized to on entry.
  //      The matrix "chart" has already been initialized to be
  //      of the correct size.
  //
  //      You can assume there are no skip arcs, i.e.,
  //      arc.get_gmm() >= 0 for all arcs "arc" in the graph.
  //      Log probabilities should be base e, i.e., natural
  //      logarithms.
  //
  //      Here is an example of the syntax for accessing the contents
  //      of a chart cell:
  //
  //      logProb = chart(frmIdx, stateIdx).get_forw_log_prob();
  //      logProb = chart(frmIdx, stateIdx).get_back_log_prob();
  //
  //      Here is an example of setting the contents of a chart cell:
  //
  //      chart(frmIdx, stateIdx).set_forw_log_prob(logProb);
  //      chart(frmIdx, stateIdx).set_back_log_prob(logProb);
  //
  //  Fill in forward pass here.

  // Init chart
  // Recursive forward pass 

  // DEBUG forward
  // cout << "forward" << endl;
  // for (int frmIdx = 0; frmIdx <= frmCnt; ++frmIdx) {
  //   for (int srcIdx = 0; srcIdx < stateCnt; ++srcIdx) {
  //     cout << format(" %d") % chart(frmIdx, srcIdx).get_forw_log_prob();
  //   }
  //   cout << endl;
  // }
  //  END_LAB
  //

  //  This function computes the total forward prob of the entire utterence,
  //  i.e., the sum of the probabilities of all complete paths through
  //  the HMM, and places it in "uttLogProb".
  //  In addition, the backward log prob of all cells for the last
  //  frame are initialized to the correct value.
  double uttLogProb = init_backward_pass(graph, chart);
  if (uttLogProb == g_zeroLogProb) return uttLogProb;

  //  BEGIN_LAB
  //
  //  Output:
  //      On exit, chart(frmIdx, stateIdx).get_back_log_prob()
  //      should be set to the appropriate backward probability.
  //      If a cell is unreachable from the start state,
  //      this value can be set to "g_zeroLogProb", which is
  //      what these values are initialized to on entry.
  //      These values have already been set correctly
  //      for the last frame, i.e., the row with index "frmCnt".
  //      The total forward log prob of the utterance can be
  //      found in "uttLogProb".
  //
  //      In addition, you need to record each (non-zero) posterior prob
  //      for each arc for each frame on the list "gmmCountList" by
  //      doing a call like:
  //
  //      gmmCountList.push_back(GmmCount(arc.get_gmm(), frmIdx,
  //          arcPosterior));
  //
  //      These counts will later be used to update GMM statistics.
  //
  //  Fill in backward pass here.
  for (int frmIdx = frmCnt-1; frmIdx >= 0; frmIdx--) {
    vector<vector<double> > backwardProbs(stateCnt);
    for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
      int arcCnt = graph.get_arc_count(stateIdx);
      int arcId = graph.get_first_arc_id(stateIdx);
      for (int arcIdx = 0; arcIdx < arcCnt; ++arcIdx) {
        Arc arc;
        arcId = graph.get_arc(arcId, arc);
        int dstState = arc.get_dst_state();
        int gmmIdx = arc.get_gmm();
        assert(gmmIdx >= 0);
        double gmmProb = gmmProbs(frmIdx, gmmIdx);
        double arcProb = arc.get_log_prob();
        double logProb = chart(frmIdx+1, dstState).get_back_log_prob();
        // Next state is activated backward
        if (logProb > g_zeroLogProb) {
          double totalProb = logProb + gmmProb + arcProb;
          backwardProbs[stateIdx].push_back(totalProb);
        }
      }
    }
    for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
      if (backwardProbs[stateIdx].size() > 0) {
        chart(frmIdx, stateIdx).set_back_log_prob(add_log_probs(backwardProbs[stateIdx])); 
      }
    }
  }
  // Recursive backward pass (Don't need terminate step)

  // DEBUG backward
  // cout << "backward\na\nb\nc" << endl;
  // for (int frmIdx = 0; frmIdx <= frmCnt; ++frmIdx) {
  //   for (int srcIdx = 0; srcIdx < stateCnt; ++srcIdx) {
  //     cout << format(" %d") % chart(frmIdx, srcIdx).get_back_log_prob();
  //   }
  //   cout << endl;
  // }
  
  // Record posterior prob
  //  END_LAB

  return uttLogProb;
}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Routine to initialize backward pass.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
double init_backward_pass(const Graph& graph, matrix<FbCell>& chart) {
  vector<double> logAddBuf;
  vector<int> finalStates;
  int frmCnt = chart.size1() - 1;
  int finalCnt = graph.get_final_state_list(finalStates);
  for (int finalIdx = 0; finalIdx < finalCnt; ++finalIdx) {
    int stateIdx = finalStates[finalIdx];
    FbCell& curCell = chart(frmCnt, stateIdx);
    if (curCell.get_forw_log_prob() == g_zeroLogProb) continue;
    double curLogProb =
        curCell.get_forw_log_prob() + graph.get_final_log_prob(stateIdx);
    logAddBuf.push_back(curLogProb);
    curCell.set_back_log_prob(graph.get_final_log_prob(stateIdx));
  }
  if (logAddBuf.empty()) {
    cout << "  No complete path found." << endl;
    return g_zeroLogProb;
  }
  return add_log_probs(logAddBuf);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

Lab2FbMain::Lab2FbMain(const map<string, string>& params)
    : m_params(params),
      m_frontEnd(m_params),
      m_gmmSet(get_required_string_param(m_params, "in_gmm")),
      m_outGmmFile(get_required_string_param(m_params, "out_gmm")),
      m_transCountsFile(get_string_param(params, "trans_counts")),
      m_iterCnt(get_int_param(m_params, "iters", 1)),
      m_iterIdx(1),
      m_totFrmCnt(0),
      m_totLogProb(0.0) {
  if (!m_transCountsFile.empty())
    m_graph.read_word_sym_table(
        get_required_string_param(params, "trans_syms"));
}

bool Lab2FbMain::init_iter() {
  if (m_iterIdx > m_iterCnt) return false;
  m_transCounts.clear();
  m_audioStrm.clear();
  m_audioStrm.open(get_required_string_param(m_params, "audio_file").c_str());
  m_graphStrm.clear();
  m_graphStrm.open(get_required_string_param(m_params, "graph_file").c_str());
  m_totFrmCnt = 0;
  m_totLogProb = 0.0;
  return true;
}

void Lab2FbMain::finish_iter() {
  m_audioStrm.close();
  m_graphStrm.close();
  cout << format("Iteration %d: %.6f logprob/frame (%d frames)") % m_iterIdx %
              (m_totFrmCnt ? m_totLogProb / m_totFrmCnt : 0.0) % m_totFrmCnt
       << endl;
  ++m_iterIdx;
}

bool Lab2FbMain::init_utt() {
  if (m_audioStrm.peek() == EOF) return false;

  m_idStr = read_float_matrix(m_audioStrm, m_inAudio);
  cout << "Processing utterance ID: " << m_idStr << endl;
  m_frontEnd.get_feats(m_inAudio, m_feats);
  if (m_feats.size2() != m_gmmSet.get_dim_count())
    throw runtime_error("Mismatch in GMM and feat dim.");
  if (m_graphStrm.peek() == EOF)
    throw runtime_error("Mismatch in number of audio files and FSM's.");
  m_graph.read(m_graphStrm, m_idStr);
  if (m_graph.get_gmm_count() > m_gmmSet.get_gmm_count())
    throw runtime_error(
        "Mismatch in number of GMM's between "
        "FSM and GmmSet.");
  m_gmmSet.calc_gmm_probs(m_feats, m_gmmProbs);

  //  Initialize dynamic programming chart.
  m_chart.resize(m_feats.size1() + 1, m_graph.get_state_count());
  m_chart.clear();
  if (m_graph.get_start_state() < 0)
    throw runtime_error("Graph has no start state.");
  m_gmmCountList.clear();
  return true;
}

void Lab2FbMain::finish_utt(double logProb) {
  m_totFrmCnt += m_feats.size1();
  m_totLogProb += logProb;
  double minPosterior = get_float_param(m_params, "min_posterior", 0.001);
  if (minPosterior > 0.0) {
    m_gmmCountListThresh.clear();
    for (int cntIdx = 0; cntIdx < (int)m_gmmCountList.size(); ++cntIdx) {
      if (m_gmmCountList[cntIdx].get_count() >= minPosterior)
        m_gmmCountListThresh.push_back(m_gmmCountList[cntIdx]);
    }
    m_gmmCountList.swap(m_gmmCountListThresh);
  }
  sort(m_gmmCountList.begin(), m_gmmCountList.end());

  string chartFile = get_string_param(m_params, "chart_file");
  if (!chartFile.empty()) {
    //  Output DP chart, for debugging.
    ofstream chartStrm(chartFile.c_str());
    int frmCnt = m_feats.size1();
    int stateCnt = m_graph.get_state_count();
    matrix<double> matForwProbs(frmCnt + 1, stateCnt);
    matrix<double> matBackProbs(frmCnt + 1, stateCnt);
    for (int frmIdx = 0; frmIdx <= frmCnt; ++frmIdx) {
      for (int srcIdx = 0; srcIdx < stateCnt; ++srcIdx) {
        matForwProbs(frmIdx, srcIdx) =
            m_chart(frmIdx, srcIdx).get_forw_log_prob();
        matBackProbs(frmIdx, srcIdx) =
            m_chart(frmIdx, srcIdx).get_back_log_prob();
      }
    }
    write_float_matrix(chartStrm, matForwProbs, m_idStr + "_forw");
    write_float_matrix(chartStrm, matBackProbs, m_idStr + "_back");

    matrix<double> matPost(frmCnt, m_gmmSet.get_gmm_count());
    matPost.clear();
    int gmmCountCnt = m_gmmCountList.size();
    for (int cntIdx = 0; cntIdx < gmmCountCnt; ++cntIdx) {
      const GmmCount& gmmCount = m_gmmCountList[cntIdx];
      matPost(gmmCount.get_frame_index(), gmmCount.get_gmm_index()) +=
          gmmCount.get_count();
    }
    write_float_matrix(chartStrm, matPost, m_idStr + "_post");
    chartStrm.close();
  }
}

void Lab2FbMain::finish() {
  m_gmmSet.write(m_outGmmFile);
  if (!m_transCountsFile.empty()) {
    ofstream countStrm(m_transCountsFile.c_str());
    for (map<int, double>::const_iterator elemIter = m_transCounts.begin();
         elemIter != m_transCounts.end(); ++elemIter)
      countStrm << format("%s %.3f\n") %
                       m_graph.get_word_sym_table().get_str(elemIter->first) %
                       elemIter->second;
    countStrm.close();
  }
}

#ifndef NO_MAIN_LOOP

void main_loop(const char** argv) {
  map<string, string> params;
  process_cmd_line(argv, params);

  Lab2FbMain mainObj(params);
  GmmStats gmmStats(mainObj.get_gmm_set(), params);
  while (mainObj.init_iter()) {
    gmmStats.clear();
    while (mainObj.init_utt()) {
      double logProb = forward_backward(
          mainObj.get_graph(), mainObj.get_gmm_probs(), mainObj.get_chart(),
          mainObj.get_gmm_counts(), mainObj.get_trans_counts());
      mainObj.finish_utt(logProb);
      gmmStats.update(mainObj.get_gmm_counts(), mainObj.get_feats());
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
