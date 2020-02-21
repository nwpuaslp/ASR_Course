/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @file lab3_lm.H
 *   @brief Main loop for Lab 3 language model training/eval.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _LAB3_LM_H
#define _LAB3_LM_H

#include "util.H"

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Encapsulation of main loop for LM training/eval.
 *
 *   Holds global variables and has routines for initializing variables
 *   and updating them for each utterance.
 *   We do this so that we can call this code from Java as well.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class Lab3LmMain {
 public:
  /** Initialize all data given parameters. **/
  Lab3LmMain(const map<string, string>& params, const SymbolTable& symTable,
             int n, int bosIdx, int eosIdx, int unkIdx);

  /** Called at the beginning of processing each utterance.
   *   Returns whether at EOF.
   **/
  bool init_utt();

  /** Called at the end of processing each utterance. **/
  void finish_utt();

  /** Called at the beginning of processing each word.
   *   Returns whether at end of sentence.
   **/
  bool init_word();

  /** Called at the end of processing each word. **/
  void finish_word(double curProb);

  /** Called at end of program. **/
  void finish();

  /** Returns current n-gram. **/
  const vector<int>& get_ngram() const { return m_ngramBuf; }

 private:
  /** Program parameters. **/
  map<string, string> m_params;

  /** The LM vocab. **/
  const SymbolTable& m_symTable;

  /** Value of "n" for the LM, i.e., as in an n-gram model. **/
  int m_n;

  /** Index of beginning-of-sentence token in vocab. **/
  int m_bosIdx;

  /** Index of end-of-sentence token in vocab. **/
  int m_eosIdx;

  /** Index of unknown token in vocab. **/
  int m_unkIdx;

  /** Stream for reading input text. **/
  ifstream m_inStrm;

  /** File to write word probs to, if desired. **/
  string m_wordProbFile;

  /** Stream for writing word probs to. **/
  ofstream m_wordProbStrm;

  /** File to write sentence log probs to, if desired. **/
  string m_sentProbFile;

  /** Stream for writing sentence log probs to. **/
  ofstream m_sentProbStrm;

  /** For holding string representation of current sentence. **/
  vector<string> m_wordList;

  /** For holding int representation of current sentence. **/
  vector<int> m_wordIdxList;

  /** Buffer for holding the current n-gram. **/
  vector<int> m_ngramBuf;

  /** Current position in sentence. **/
  int m_posIdx;

  /** Total number of words processed so far. **/
  int m_totWordCnt;

  /** Total log prob of words processed so far. **/
  double m_totLogProb;

  /** Total log prob of words in current sentence processed so far. **/
  double m_sentLogProb;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#endif
