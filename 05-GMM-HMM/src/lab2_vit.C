
//  $Id: lab2_vit.C,v 1.46 2009/10/03 03:16:16 stanchen Exp $

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @file lab2_vit.C
 *   @brief This program does Viterbi decoding on a series of utterances
 *   given a decoding graph or graphs and a set of GMM's.
 *
 *   This program can be used in two ways:
 *   (1) Regular decoding, outputting the most likely word sequence
 *       for each utterance.  In this case, the same decoding graph
 *       is used across all utterances.
 *   (2) Forced alignment, outputting the most likely GMM sequence
 *       for each utterance.  This is useful for Viterbi-style
 *       training.  In this case, a different graph is used for
 *       each utterance, corresponding to the expansion of the
 *       reference transcript for that utterance.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <cassert>
#include "front_end.H"  
#include "lab2_vit.H"
#include "util.H"

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Routine for Viterbi decoding.
 *
 *   @param graph HMM/graph to operate on.
 *   @param gmmProbs Matrix of log prob for each GMM for each frame.
 *   @param chart Dynamic programming chart to fill in; already
 *       allocated to be of correct size and initialized with default values.
 *   @param outLabelList Indices of decoded output tokens are placed here.
 *   @param acousWgt Acoustic weight.
 *   @param doAlign If true, return GMM indices rather than word indices
 *       in @p outLabelList.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
double viterbi(const Graph& graph, const matrix<double>& gmmProbs,
               matrix<VitCell>& chart, vector<int>& outLabelList,
               double acousWgt, bool doAlign) {
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
  //      A matrix "chart" of "VitCell" objects (declaration of
  //      "VitCell" class above):
  //
  //      chart(0 .. frmCnt, 0 .. stateCnt - 1)
  //
  //      On exit, chart(frmIdx, stateIdx).get_log_prob()
  //      should be set to the logarithm of the probability
  //      of the best path from the start state to
  //      state "stateIdx" given the
  //      first "frmIdx" frames of observations;
  //      and chart(frmIdx, stateIdx).get_arc_id() should be set
  //      to the arc ID for the last arc of this best path (or -1
  //      if the best path is of length 0).
  //      If a cell is unreachable from the start state,
  //      these values should be set to "g_zeroLogProb"
  //      and -1, respectively, which are what these values
  //      are initialized to on entry.
  //      The matrix "chart" has already been initialized to be
  //      of the correct size.
  //
  //      Notes: "g_zeroLogProb" is a large negative number we use
  //      to represent "ln 0" instead of the actual
  //      value negative infinity.
  //      You can assume there are no skip arcs, i.e.,
  //      arc.get_gmm() >= 0 for all arcs "arc" in the graph.
  //      Log probabilities should be base e, i.e., natural
  //      logarithms.
  //
  //      Here is an example of the syntax for accessing a chart
  //      cell log prob:
  //
  //      logProb = chart(frmIdx, stateIdx).get_log_prob();
  //
  //      Here is an example of setting the contents of a chart cell:
  //
  //      chart(frmIdx, stateIdx).assign(logProb, arcId);
  //
  //  Fill in Viterbi algorithm here.
  //
  //  The code for calculating the final probability and
  //  the best path is provided for you below.
  // assert(graph.get_state_count() == stateCnt);

  // DEBUG chart BEGIN
  // int frmMax = frmCnt + 1;
  // for (int frmIdx = 0; frmIdx < frmMax; ++frmIdx) {
  //   // log prob
  //   for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
  //     cout << format(" %d") % chart(frmIdx, stateIdx).get_log_prob();
  //   }
  //   cout << endl;
  // }
  // // arc id
  // for (int frmIdx = 0; frmIdx < frmMax; ++frmIdx) {
  //   for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
  //     cout << format(" %d") % chart(frmIdx, stateIdx).get_arc_id();
  //   }
  //   cout << endl;
  // }
  // DEBUG chart END
  // return 0.0;

  //  END_LAB
  //

  //  The code for calculating the final probability and
  //  the best path is provided for you.
  return viterbi_backtrace(graph, chart, outLabelList, doAlign);
}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Routine for Viterbi backtrace.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
double viterbi_backtrace(const Graph& graph, matrix<VitCell>& chart,
                         vector<int>& outLabelList, bool doAlign) {
  int frmCnt = chart.size1() - 1;
  int stateCnt = chart.size2();

  //  Find best final state.
  vector<int> finalStates;
  int finalCnt = graph.get_final_state_list(finalStates);
  double bestLogProb = g_zeroLogProb;
  int bestFinalState = -1;
  for (int finalIdx = 0; finalIdx < finalCnt; ++finalIdx) {
    int stateIdx = finalStates[finalIdx];
    if (chart(frmCnt, stateIdx).get_log_prob() == g_zeroLogProb) continue;
    double curLogProb = chart(frmCnt, stateIdx).get_log_prob() +
                        graph.get_final_log_prob(stateIdx);
    if (curLogProb > bestLogProb)
      bestLogProb = curLogProb, bestFinalState = stateIdx;
  }
  if (bestFinalState < 0) throw runtime_error("No complete paths found.");

  //  Do backtrace, collect appropriate labels.
  outLabelList.clear();
  int stateIdx = bestFinalState;
  // cout << format("frmCnt %d\n") % frmCnt;
  for (int frmIdx = frmCnt; --frmIdx >= 0;) {
    assert((stateIdx >= 0) && (stateIdx < stateCnt));
    int arcId = chart(frmIdx + 1, stateIdx).get_arc_id();
    Arc arc;
    // cout << format("frmIdx: %d arcId: %d") % frmIdx % arcId << endl;
    graph.get_arc(arcId, arc);
    assert((int)arc.get_dst_state() == stateIdx);
    // cout << "HERE" << endl;
    if (doAlign) {
      if (arc.get_gmm() < 0)
        throw runtime_error("Expect all arcs to have GMM.");
      outLabelList.push_back(arc.get_gmm());
    } else if (arc.get_word() > 0)
      outLabelList.push_back(arc.get_word());
    stateIdx = graph.get_src_state(arcId);
  }
  if (stateIdx != graph.get_start_state())
    throw runtime_error("Backtrace does not end at start state.");
  reverse(outLabelList.begin(), outLabelList.end());
  return bestLogProb;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

Lab2VitMain::Lab2VitMain(const map<string, string>& params)
    : m_params(params),
      m_frontEnd(m_params),
      m_gmmSet(get_required_string_param(m_params, "gmm")),
      m_audioStrm(get_required_string_param(m_params, "audio_file").c_str()),
      m_doAlign(!get_string_param(m_params, "align_file").empty()),
      m_outStrm(get_required_string_param(
                    m_params, !m_doAlign ? "dcd_file" : "align_file")
                    .c_str()),
      m_acousWgt(get_float_param(m_params, "ac_wgt", 1.0)),
      m_totFrmCnt(0),
      m_totLogProb(0.0) {
  if (!m_doAlign)
    m_graph.read(get_required_string_param(m_params, "graph_file"),
                 get_required_string_param(m_params, "word_syms"));
  else
    m_graphStrm.open(get_required_string_param(m_params, "graph_file").c_str());
}

bool Lab2VitMain::init_utt() {
  if (m_audioStrm.peek() == EOF) return false;

  m_idStr = read_float_matrix(m_audioStrm, m_inAudio);
  cout << "Processing utterance ID: " << m_idStr << endl;
  m_frontEnd.get_feats(m_inAudio, m_feats);
  if (m_feats.size2() != m_gmmSet.get_dim_count())
    throw runtime_error("Mismatch in GMM and feat dim.");
  if (m_doAlign) {
    if (m_graphStrm.peek() == EOF)
      throw runtime_error(
          "Mismatch in number of audio files "
          "and FSM's.");
    m_graph.read(m_graphStrm, m_idStr);
  }
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
  return true;
}

void Lab2VitMain::finish_utt(double logProb) {
  m_totFrmCnt += m_feats.size1();
  m_totLogProb += logProb;

  //  Output results.
  if (m_doAlign)
    write_int_vector(m_outStrm, m_labelList, m_idStr);
  else {
    cout << "  Output:";
    for (int labelIdx = 0; labelIdx < (int)m_labelList.size(); ++labelIdx) {
      m_outStrm << m_graph.get_word_sym_table().get_str(m_labelList[labelIdx])
                << " ";
      cout << " "
           << m_graph.get_word_sym_table().get_str(m_labelList[labelIdx]);
    }
    m_outStrm << "(" << m_idStr << ")" << endl;
    cout << endl;
  }

  string chartFile = get_string_param(m_params, "chart_file");
  if (!chartFile.empty()) {
    //  Output DP chart, for debugging.
    ofstream chartStrm(chartFile.c_str());
    int frmCnt = m_feats.size1();
    int stateCnt = m_graph.get_state_count();
    matrix<double> matProbs(frmCnt + 1, stateCnt);
    matrix<int> matArcs(frmCnt + 1, stateCnt);
    for (int frmIdx = 0; frmIdx <= frmCnt; ++frmIdx) {
      for (int srcIdx = 0; srcIdx < stateCnt; ++srcIdx) {
        matProbs(frmIdx, srcIdx) = m_chart(frmIdx, srcIdx).get_log_prob();
        matArcs(frmIdx, srcIdx) = m_chart(frmIdx, srcIdx).get_arc_id();
      }
    }
    write_float_matrix(chartStrm, matProbs, m_idStr + "_probs");
    write_int_matrix(chartStrm, matArcs, m_idStr + "_arcs");
    chartStrm.close();
  }
}

void Lab2VitMain::finish() {
  m_audioStrm.close();
  if (m_doAlign) m_graphStrm.close();
  m_outStrm.close();
  cout << format("%.6f logprob/frame (%d frames).") %
              (m_totFrmCnt ? m_totLogProb / m_totFrmCnt : 0.0) % m_totFrmCnt
       << endl;
}

#ifndef NO_MAIN_LOOP

void main_loop(const char** argv) {
  map<string, string> params;
  process_cmd_line(argv, params);

  Lab2VitMain mainObj(params);
  while (mainObj.init_utt()) {
    double logProb = viterbi(mainObj.get_graph(), mainObj.get_gmm_probs(),
                             mainObj.get_chart(), mainObj.get_label_list(),
                             mainObj.get_acous_wgt(), mainObj.do_align());
    mainObj.finish_utt(logProb);
  }
  mainObj.finish();
}

#endif

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
