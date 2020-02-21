#include <set>
#include "lang_model.H"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

LangModel::LangModel(const map<string, string>& params)
    : m_params(params),
      m_symTable(new SymbolTable(get_required_string_param(m_params, "vocab"))),
      m_bosIdx(m_symTable->get_index(get_string_param(m_params, "bos", "<s>"))),
      m_eosIdx(
          m_symTable->get_index(get_string_param(m_params, "eos", "</s>"))),
      m_unkIdx(
          m_symTable->get_index(get_string_param(m_params, "unk", "<UNK>"))),
      m_n(get_int_param(m_params, "n", 3)) {
  if ((m_bosIdx == -1) || (m_eosIdx == -1) || (m_unkIdx == -1))
    throw runtime_error("Vocabulary missing BOS/EOS/UNK token.");

  //  Get training counts.
  ifstream inStrm(get_required_string_param(params, "train").c_str());
  string lineStr;
  vector<string> wordList;
  vector<int> wordIdxList;
  while (inStrm.peek() != EOF) {
    //  Read line, split into tokens, convert to indices, count n-grams.
    getline(inStrm, lineStr);
    split_string(lineStr, wordList);
    convert_words_to_indices(wordList, wordIdxList, get_sym_table(),
                             get_ngram_length(), get_bos_index(),
                             get_eos_index(), get_unknown_index());
    count_sentence_ngrams(wordIdxList);
  }

  string countFile = get_string_param(params, "count_file");
  if (!countFile.empty()) write_counts(countFile);
}

void LangModel::write_counts(const string& fileName) const {
  ofstream outStrm(fileName.c_str());
  outStrm << "# Pred counts.\n";
  m_predCounts.write(outStrm, get_sym_table());
  outStrm << "# Hist counts.\n";
  m_histCounts.write(outStrm, get_sym_table());
  outStrm << "# Hist 1+ counts.\n";
  m_histOnePlusCounts.write(outStrm, get_sym_table());
  outStrm.close();
}

void LangModel::count_sentence_ngrams(const vector<int>& wordList) {
  int wordCnt = wordList.size();

  //
  //  BEGIN_LAB
  //
  //  This routine is called for each sentence in the training
  //  data.  It should collect all relevant n-gram counts for
  //  the sentence.
  //
  //  Input:
  //      "m_n" = the value of "n" for the n-gram model; e.g.,
  //          this has the value 3 for a trigram model.
  //      "wordCnt" = the number of words in the vector "wordList".
  //
  //      The vector "wordList[0 .. (wordCnt-1)]" holds the sentence
  //      as a sequence of integer indices (each integer
  //      representing a word).  This vector is padded with
  //      beginning-of-sentence and end-of-sentence markers in
  //      a convenient manner for counting.  In particular,
  //      "wordList[0 .. (m_n - 2)]" holds beginning-of-sentence markers, so
  //      the first "real" word of the sentence is "wordList[m_n - 1]";
  //      and "wordList[wordCnt-1]" holds the end-of-sentence marker.
  //
  //  Output:
  //      Update all the counts needed for Witten-Bell smoothing.
  //
  //      Specifically, the objects "m_predCounts", "m_histCounts", and
  //      "m_histOnePlusCounts" are all of type "NGramCounter", a class
  //      that can be used to store counts for a set of n-grams.
  //      The object "m_predCounts" should be used to store
  //      "regular" n-gram counts; the object "m_histCounts" should
  //      be used to store "history" n-gram counts (i.e., the
  //      normalization count for that n-gram occuring as a history);
  //      and "m_histOnePlusCounts" should be used to store the
  //      number of different words following that n-gram history.
  //      To increment the count of a particular n-gram in
  //      one of these objects, you can do a call like:
  //
  //      m_predCounts.incr_count(ngram);
  //
  //      where "ngram" is of type "vector<int>".  This call returns
  //      the value of the incremented count.
  //
  //      Your code should work for any value of m_n (larger than zero).
  assert(m_n > 0);

  // static map<vector<int>, set<int> > histOnePlusMap;
  for (int wordIdx = m_n - 1; wordIdx < wordCnt; ++wordIdx) {
    // process from n-gram to 1-gram
    for (int n = 1; n <= m_n; ++n) {
      // process m_predCounts
      vector<int> ngram(wordList.begin() + wordIdx - (m_n - n),
                        wordList.begin() + wordIdx + 1);
      int count = m_predCounts.incr_count(ngram);
      // process m_histCounts
      vector<int> histNgram(ngram.begin(), ngram.end() - 1);
      m_histCounts.incr_count(histNgram);
      // process m_histOnePlusCounts
      // histOnePlusMap[histNgram].insert(*(ngram.end() - 1));
      // m_histOnePlusCounts.set_count(histNgram,
      //                               histOnePlusMap[histNgram].size());
      // another implementation of m_histOnePlusCounts:
      if (count == 1) {
        m_histOnePlusCounts.incr_count(histNgram);
      }
    }
  }

  //  END_LAB
  //
}

double LangModel::get_prob_witten_bell(const vector<int>& ngram) const {
  double retProb = 1.0;
  //  Don't count epsilon.
  int vocSize = m_symTable->size() - 1;

  //
  //  BEGIN_LAB
  //
  //  This routine should return an n-gram probability smoothed
  //  with Witten-Bell smoothing.
  //
  //  Input:
  //      "m_n" = the value of "n" for the n-gram model.
  //      "ngram" = the input n-gram, expressed as a vector of integers.
  //          This will be of length "m_n".
  //      "m_predCounts" = object holding "regular" counts for each n-gram.
  //      "m_histCounts" = object holding "history" counts for each n-gram.
  //      "m_histOnePlusCounts" = object holding number of unique words
  //          following each n-gram history in the training data.
  //      "vocSize" = vocabulary size.
  //
  //      To access the count of an n-gram, you can do something like:
  //
  //      int predCnt = m_predCounts.get_count(ngram);
  //
  //      where "ngram" is of type "vector<int>".
  //
  //  Output:
  //      "retProb" should be set to the smoothed n-gram probability
  //          of the last word in the n-gram given the previous words.
  //

  vector<int> histNgram(ngram.begin(), ngram.end() - 1);
  int predCnt = m_predCounts.get_count(ngram);
  int histCnt = m_histCounts.get_count(histNgram);
  int histOnePlusCnt = m_histOnePlusCounts.get_count(histNgram);

  double lambda = 0.0, PMle = 0.0, beta = 1.0, PBackoff;
  if (histCnt > 0) {
    lambda = 1.0 * histCnt / (histCnt + histOnePlusCnt);
    PMle = 1.0 * predCnt / histCnt;
    beta = 1.0 * histOnePlusCnt / (histCnt + histOnePlusCnt);
  }
  if (ngram.size() == 1) {
    // recursive terminate
    PBackoff = 1.0 / vocSize;
  } else {
    // recursive
    PBackoff =
        get_prob_witten_bell(vector<int>(ngram.begin() + 1, ngram.end()));
  }
  retProb = lambda * PMle + beta * PBackoff;

  //  END_LAB
  //

  return retProb;
}

double LangModel::get_prob(const vector<int>& ngram) const {
  if ((ngram.size() < 1) || ((int)ngram.size() > m_n))
    throw runtime_error("Invalid n-gram size.");
  return get_prob_witten_bell(ngram);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
