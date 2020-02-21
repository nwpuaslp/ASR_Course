/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @file lang_model.H
 *   @brief Contains LangModel class.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _LANG_MODEL_H
#define _LANG_MODEL_H

#include "util.H"

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Language model class.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class LangModel {
 public:
  /** Ctor; get configuration parameters from "params". **/
  LangModel(const map<string, string>& params = ParamsType());

  /** Get associated mapping from word spellings to integer indices. **/
  const SymbolTable& get_sym_table() const { return *m_symTable.get(); }

  /** Returns index of beginning-of-sentence token. **/
  int get_bos_index() const { return m_bosIdx; }

  /** Returns index of end-of-sentence token. **/
  int get_eos_index() const { return m_eosIdx; }

  /** Returns index of unknown token. **/
  int get_unknown_index() const { return m_unkIdx; }

  /** Returns "n" of n-gram. **/
  int get_ngram_length() const { return m_n; }

  /** Returns conditional prob of last word given previous words.
   *   The argument @p ngram can be any length from 1 to the
   *   value of #get_ngram_length().
   **/
  double get_prob(const vector<int>& ngram) const;

 private:
  /** Count n-grams in a training sentence. **/
  void count_sentence_ngrams(const vector<int>& wordIdxList);

  /** Returns conditional prob of last word given previous words
   *   under Witten-Bell smoothing.
   *   The argument @p ngram can be any length from 1 to the
   *   value of #get_ngram_length().
   **/
  double get_prob_witten_bell(const vector<int>& ngram) const;

  /** Write out all counts to a file, for debugging. **/
  void write_counts(const string& fileName) const;

 private:
  /** Stores copy of input parameters. **/
  map<string, string> m_params;

  /** Map from words to integer indices. **/
  shared_ptr<SymbolTable> m_symTable;

  /** Index of beginning-of-sentence token. **/
  int m_bosIdx;

  /** Index of end-of-sentence token. **/
  int m_eosIdx;

  /** Index of unknown token. **/
  int m_unkIdx;

  /** Value of "n". **/
  int m_n;

  /** Map from pred n-grams to their counts. **/
  NGramCounter m_predCounts;

  /** Map from hist n-grams to their counts. **/
  NGramCounter m_histCounts;

  /** Map from hist n-grams to their "1+" counts. **/
  NGramCounter m_histOnePlusCounts;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#endif
