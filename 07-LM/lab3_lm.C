/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @file lab3_lm.C
 *   @brief This program constructs/loads a language model and
 *   evaluates the perplexity of a sequence of input sentences.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include "lab3_lm.H"
#include "lang_model.H"
#include "util.H"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

Lab3LmMain::Lab3LmMain(const map<string, string>& params,
                       const SymbolTable& symTable, int n, int bosIdx,
                       int eosIdx, int unkIdx)
    : m_params(params),
      m_symTable(symTable),
      m_n(n),
      m_bosIdx(bosIdx),
      m_eosIdx(eosIdx),
      m_unkIdx(unkIdx),
      m_inStrm(get_required_string_param(m_params, "test").c_str()),
      m_wordProbFile(get_string_param(m_params, "word_probs")),
      m_sentProbFile(get_string_param(m_params, "sent_log_probs")),
      m_posIdx(0),
      m_totWordCnt(0),
      m_totLogProb(0.0),
      m_sentLogProb(0.0) {
  if (!m_wordProbFile.empty()) m_wordProbStrm.open(m_wordProbFile.c_str());
  if (!m_sentProbFile.empty()) m_sentProbStrm.open(m_sentProbFile.c_str());
}

bool Lab3LmMain::init_utt() {
  if (m_inStrm.peek() == EOF) return false;

  //  Read line, split into tokens, convert to indices.
  string lineStr;
  getline(m_inStrm, lineStr);
  split_string(lineStr, m_wordList);
  convert_words_to_indices(m_wordList, m_wordIdxList, m_symTable, m_n, m_bosIdx,
                           m_eosIdx, m_unkIdx);
  m_posIdx = m_n - 1;
  m_sentLogProb = 0.0;
  return true;
}

void Lab3LmMain::finish_utt() {
  if (!m_sentProbFile.empty())
    m_sentProbStrm << format("%.6f\n") % m_sentLogProb;
}

bool Lab3LmMain::init_word() {
  if (m_posIdx >= (int)m_wordIdxList.size()) return false;
  m_ngramBuf.clear();
  m_ngramBuf.insert(m_ngramBuf.end(),
                    m_wordIdxList.begin() + m_posIdx - m_n + 1,
                    m_wordIdxList.begin() + m_posIdx + 1);
  return true;
}

void Lab3LmMain::finish_word(double curProb) {
  ++m_posIdx;
  ++m_totWordCnt;
  m_totLogProb += (curProb > 0.0) ? log(curProb) : g_zeroLogProb;
  m_sentLogProb += (curProb > 0.0) ? log(curProb) : g_zeroLogProb;
  if (!m_wordProbFile.empty()) m_wordProbStrm << format("%.6f\n") % curProb;
}

void Lab3LmMain::finish() {
  m_inStrm.close();
  if (!m_wordProbFile.empty()) m_wordProbStrm.close();
  if (!m_sentProbFile.empty()) m_sentProbStrm.close();
  cout << format("%.4f PP (%d words)") %
              exp(m_totWordCnt ? -m_totLogProb / m_totWordCnt : 0.0) %
              m_totWordCnt
       << endl;
}

#ifndef NO_MAIN_LOOP

void main_loop(const char** argv) {
  map<string, string> params;
  process_cmd_line(argv, params);

  //  Initialize language model.
  LangModel lm(params);

  Lab3LmMain mainObj(params, lm.get_sym_table(), lm.get_ngram_length(),
                     lm.get_bos_index(), lm.get_eos_index(),
                     lm.get_unknown_index());
  while (mainObj.init_utt()) {
    while (mainObj.init_word()) {
      double curProb = lm.get_prob(mainObj.get_ngram());
      mainObj.finish_word(curProb);
    }
    mainObj.finish_utt();
  }
  mainObj.finish();
}

#endif

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
