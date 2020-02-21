/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @file util.H
 *   @brief I/O routines, GmmSet, and Graph classes.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef _UTIL_H
#define _UTIL_H

#include <algorithm>
#include <boost/format.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/shared_ptr.hpp>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

#ifndef SWIG
using boost::format;
using boost::numeric::ublas::matrix;
using boost::str;
// using boost::shared_ptr;
using std::shared_ptr;
#endif

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @name Math stuff.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

//@{

/** This value can be used to represent the logprob of a "zero" prob.
 *   Theoretically, log 0 is negative infinity which we can't store,
 *   so we can use this very large negative value instead.
 **/
const double g_zeroLogProb = -FLT_MAX / 2.0;

/** Adds the log probs held in @p logProbList, returning answer as log prob.
 *   That is, let's say we have a list of probability values, the logs of
 *   which are stored in @p logProbList.  Then, this routine returns the
 *   log of the sum of those probability values.
 *   Logarithms are base <i>e</i>.
 **/
double add_log_probs(const vector<double>& logProbList);

/** Does in-place real FFT.
 *   For inputs <tt>vals[i]</tt>, i = 0, ..., N-1 with sample period T,
 *   on return the real and imaginary parts of the FFT value for frequency
 *   i/NT are held in the outputs <tt>vals[2*i]</tt> and <tt>vals[2*i+1]</tt>.
 **/
void real_fft(vector<double>& vals);

/** Sets @p vec to be equal to the @p rowIdx-th row of @p mat.
 *   Rows are numbered starting from 0.
 **/
void copy_matrix_row_to_vector(const matrix<double>& mat, unsigned rowIdx,
                               vector<double>& vec);

/** Sets the @p rowIdx-th row of @p mat to @p vec; sizes must match.
 *   Rows are numbered starting from 0.
 **/
void copy_vector_to_matrix_row(const vector<double>& vec, matrix<double>& mat,
                               unsigned rowIdx);

//@}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @name Command-line parsing and parameter lookup routines.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

//@{

/** Type of object used for holding program parameters.
 *   Declaration needed for hack to get default arguments to work.
 **/
typedef map<string, string> ParamsType;

#ifndef SWIG
/** Given cmd line arguments @p argv, parses flags of the
 *   form <tt>--&lt;flag&gt; &lt;val&gt;</tt> and places in flag-to-value
 *   map @p params.
 *   Expects same @p argv value as passed to <tt>main()</tt>.
 *   Existing values in @p params are not erased (unless overriden
 *   in @p argv).
 **/
void process_cmd_line(const char** argv, map<string, string>& params);
#endif

/** Like process_cmd_line(), but expects arguments as string vector. **/
void process_cmd_line(const vector<string>& argList,
                      map<string, string>& params);

/** Like process_cmd_line(), but expects space-separated arguments
 *   in single string.
 **/
void process_cmd_line(const string& argStr, map<string, string>& params);

/** Returns value of boolean parameter @p name from parameter map @p params.
 *   If not present, returns @p defaultVal.
 **/
bool get_bool_param(const map<string, string>& params, const string& name,
                    bool defaultVal = false);

/** Like get_bool_param(), but for integer parameters. **/
int get_int_param(const map<string, string>& params, const string& name,
                  int defaultVal = 0);

/** Like get_bool_param(), but for floating-point parameters. **/
double get_float_param(const map<string, string>& params, const string& name,
                       double defaultVal = 0.0);

/** Like get_bool_param(), but for string parameters. **/
string get_string_param(const map<string, string>& params, const string& name,
                        const string& defaultVal = string());

/** Like get_string_param(), but throws exception if parameter absent. **/
string get_required_string_param(const map<string, string>& params,
                                 const string& name);

//@}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   @name Vector/matrix I/O routines.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

//@{

/** Splits @p inStr into space-separated tokens; places in @p outList. **/
void split_string(const string& inStr, vector<string>& outList);

/** Reads a list of strings, one to a line, from file @p fileName and
 *   places in @p strList.
 **/
void read_string_list(const string& fileName, vector<string>& strList);

/** Reads matrix of floating-point numbers from stream @p inStrm in Matlab
 *   text format and places in @p mat.  Expects optional matrix header, and
 *   then one row per line.  If argument @p name is provided, checks
 *   name associated with matrix matches and throws exception if doesn't.
 *   Returns name given in matrix header, or empty string if none provided.
 **/
string read_float_matrix(istream& inStrm, matrix<double>& mat,
                         const string& name = string());

/** Like read_float_matrix(), but for float vectors. **/
string read_float_vector(istream& inStrm, vector<double>& vec,
                         const string& name = string());

/** Like read_float_matrix(), but for integer matrices. **/
string read_int_matrix(istream& inStrm, matrix<int>& mat,
                       const string& name = string());

/** Like read_float_matrix(), but for integer vectors. **/
string read_int_vector(istream& inStrm, vector<int>& vec,
                       const string& name = string());

/** Like read_float_matrix(), but reads from file @p fileName instead
 *   of stream.
 **/
void read_float_matrix(const string& fileName, matrix<double>& mat);

/** Like read_float_vector(), but reads from file @p fileName instead
 *   of stream.
 **/
void read_float_vector(const string& fileName, vector<double>& vec);

/** Like read_int_matrix(), but reads from file @p fileName instead
 *   of stream.
 **/
void read_int_matrix(const string& fileName, matrix<int>& mat);

/** Like read_int_vector(), but reads from file @p fileName instead
 *   of stream.
 **/
void read_int_vector(const string& fileName, vector<int>& vec);

/** Writes floating-point matrix @p mat to stream @p outStrm in Matlab text
 *   format.  If the argument @p name is provided, this name will
 *   be written in the matrix header (and is the name
 *   that will be assigned to the matrix if loaded in octave).
 **/
void write_float_matrix(ostream& outStrm, const matrix<double>& mat,
                        const string& name = string());

/** Like write_float_matrix(), but for float vectors. **/
void write_float_vector(ostream& outStrm, const vector<double>& vec,
                        const string& name = string());

/** Like write_float_matrix(), but for integer matrices. **/
void write_int_matrix(ostream& outStrm, const matrix<int>& mat,
                      const string& name = string());

/** Like write_float_matrix(), but for integer vectors. **/
void write_int_vector(ostream& outStrm, const vector<int>& vec,
                      const string& name = string());

/** Like write_float_matrix(), but writes to file @p fileName instead
 *   of stream.
 **/
void write_float_matrix(const string& fileName, const matrix<double>& mat);

/** Like write_float_vector(), but writes to file @p fileName instead
 *   of stream.
 **/
void write_float_vector(const string& fileName, const vector<double>& vec);

/** Like write_int_matrix(), but writes to file @p fileName instead
 *   of stream.
 **/
void write_int_matrix(const string& fileName, const matrix<int>& mat);

/** Like write_int_vector(), but writes to file @p fileName instead
 *   of stream.
 **/
void write_int_vector(const string& fileName, const vector<int>& vec);

//@}

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Class holding set of diagonal covariance GMM's.
 *
 *   Here, we summarize the key routines for accessing the parameters of
 *   each GMM.  Each GMM has a set of component Gaussians.  To find the
 *   number of components of a GMM, use #get_component_count().  To find the
 *   mixture weight of a particular component of a GMM,
 *   use #get_component_weight().
 *   To get the means and variances of a particular component, first call
 *   #get_gaussian_index() to find the index of the corresponding Gaussian.
 *   Then, one can call #get_gaussian_mean() and #get_gaussian_var()
 *   with this index to find the means and variances.
 *   The reason for this indirection is to support the sharing of
 *   Gaussians between GMM's.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class GmmSet {
 public:
  /** Ctor; loads from file @p fileName if argument present. **/
  GmmSet(const string& fileName = string());

  /** Reads GMM parameters from file @p fileName. **/
  void read(const string& fileName);

  /** Writes GMM parameters to file @p fileName. **/
  void write(const string& fileName) const;

  /** Initializes object, where the vector @p gmmCompCounts holds the
   *   number of component Gaussians of each GMM, and @p dimCnt
   *   is the dimension of
   *   the Gaussians.  Component weights are initialized to be
   *   uniform; and means and variances for each dimension are set
   *   to 0 and 1, respectively.
   *   If the optional argument @p compMap is not provided, then
   *   no Gaussians are shared between GMM's.  Otherwise, the vector
   *   @p compMap holds the index of the Gaussian for each
   *   component of each GMM, in order; i.e., it contains first
   *   the indices of each component of the 0th GMM, then the
   *   1st GMM, etc.
   **/
  void init(const vector<int>& gmmCompCounts, unsigned dimCnt,
            const vector<int>& compMap = vector<int>());

  /** Clears object, i.e., deletes all GMM's. **/
  void clear();

  /** Returns whether object is empty, i.e., has no GMM's. **/
  bool empty() const { return m_gmmMap.empty(); }

  /** Returns number of GMM's in model. **/
  unsigned get_gmm_count() const { return m_gmmMap.size(); }

  /** Returns total number of individual Gaussians in model. **/
  unsigned get_gaussian_count() const { return m_gaussParams.size1(); }

  /** Returns dimension of Gaussians. **/
  unsigned get_dim_count() const {
    assert(!(m_gaussParams.size2() & 1));
    return m_gaussParams.size2() / 2;
  }

  /** Returns number of component Gaussians in @p gmmIdx-th GMM.
   *   GMM's are numbered starting from 0.
   **/
  unsigned get_component_count(unsigned gmmIdx) const {
    return get_max_comp_index(gmmIdx) - get_min_comp_index(gmmIdx);
  }

  /** Returns Gaussian index for @p compIdx-th component of
   *   @p gmmIdx-th GMM.  This index is needed for looking up
   *   or setting Gaussian parameters.
   *   GMM's and components are numbered starting from 0.
   *   @see #get_gaussian_mean(), #get_gaussian_var(), etc.
   **/
  unsigned get_gaussian_index(unsigned gmmIdx, unsigned compIdx) const {
    assert((gmmIdx < m_gmmMap.size()) &&
           (compIdx < get_component_count(gmmIdx)));
    return m_compMap[get_min_comp_index(gmmIdx) + compIdx];
  }

  /** Returns mixture weight of @p compIdx-th component of
   *   @p gmmIdx-th GMM.
   *   GMM's and components are numbered starting from 0.
   **/
  double get_component_weight(unsigned gmmIdx, unsigned compIdx) const {
    assert((gmmIdx < m_gmmMap.size()) &&
           (compIdx < get_component_count(gmmIdx)));
    return m_compWeights[get_min_comp_index(gmmIdx) + compIdx];
  }

  /** Sets mixture weight of @p compIdx-th component of
   *   @p gmmIdx-th GMM to @p wgt.
   *   GMM's and components are numbered starting from 0.
   **/
  void set_component_weight(unsigned gmmIdx, unsigned compIdx, double wgt) {
    assert((gmmIdx < m_gmmMap.size()) &&
           (compIdx < get_component_count(gmmIdx)));
    m_compWeights[get_min_comp_index(gmmIdx) + compIdx] = wgt;
    m_normsUpToDate = false;
  }

  /** Returns mean for dimension @p dimIdx for Gaussian  with index
   *   @p gaussIdx.
   *   Gaussians and dimensions are numbered starting from 0.
   **/
  double get_gaussian_mean(unsigned gaussIdx, unsigned dimIdx) const {
    assert((gaussIdx < m_gaussParams.size1()) &&
           (2 * dimIdx < m_gaussParams.size2()));
    return m_gaussParams(gaussIdx, 2 * dimIdx);
  }

  /** Returns variance for dimension @p dimIdx for Gaussian with index
   *   @p gaussIdx.
   *   Gaussians and dimensions are numbered starting from 0.
   **/
  double get_gaussian_var(unsigned gaussIdx, unsigned dimIdx) const {
    assert((gaussIdx < m_gaussParams.size1()) &&
           (2 * dimIdx + 1 < m_gaussParams.size2()));
    return m_gaussParams(gaussIdx, 2 * dimIdx + 1);
  }

  /** Sets mean for dimension @p dimIdx for Gaussian @p gaussIdx.
   *   Gaussians and dimensions are numbered starting from 0.
   **/
  void set_gaussian_mean(unsigned gaussIdx, unsigned dimIdx, double val) {
    assert((gaussIdx < m_gaussParams.size1()) &&
           (2 * dimIdx < m_gaussParams.size2()));
    m_gaussParams(gaussIdx, 2 * dimIdx) = val;
    m_normsUpToDate = false;
  }

  /** Sets variance for dimension @p dimIdx for Gaussian @p gaussIdx.
   *   Gaussians and dimensions are numbered starting from 0.
   **/
  void set_gaussian_var(unsigned gaussIdx, unsigned dimIdx, double val) {
    assert((gaussIdx < m_gaussParams.size1()) &&
           (2 * dimIdx + 1 < m_gaussParams.size2()));
    m_gaussParams(gaussIdx, 2 * dimIdx + 1) = val;
    m_normsUpToDate = false;
  }

  /** Copies the means and variances of Gaussian @p srcGaussIdx
   *   in @p srcGmmSet into Gaussian @p dstGaussIdx in this object.
   *   Gaussians are numbered starting from 0.
   **/
  void copy_gaussian(unsigned dstGaussIdx, const GmmSet& srcGmmSet,
                     unsigned srcGaussIdx);

  /** Given input feature vectors, computes log prob (base e) of each GMM
   *   for each frame; i.e., on exit @p logProbs will have the
   *   same number of rows as @p feats and one column for each GMM.
   **/
  void calc_gmm_probs(const matrix<double>& feats,
                      matrix<double>& logProbs) const;

  /** Computes log prob (base e) of each component Gaussian for
   *   GMM @p gmmIdx for feature vector @p feats.
   *   Places result in @p logProbs.  Returns total log prob of GMM.
   *   GMM's are numbered starting from 0.
   **/
  double calc_component_probs(const vector<double>& feats, unsigned gmmIdx,
                              vector<double>& logProbs) const;

 private:
  /** Returns index of first component for GMM @p gmmIdx. **/
  unsigned get_min_comp_index(unsigned gmmIdx) const {
    assert(gmmIdx < m_gmmMap.size());
    return m_gmmMap[gmmIdx];
  }

  /** Returns one past index of last component for GMM @p gmmIdx. **/
  unsigned get_max_comp_index(unsigned gmmIdx) const {
    assert(gmmIdx < m_gmmMap.size());
    return (gmmIdx + 1 < m_gmmMap.size()) ? m_gmmMap[gmmIdx + 1]
                                          : m_compMap.size();
  }

  /** Recomputes normalization constants useful for log prob
   *   computation.
   **/
  void compute_norms() const;

  /** Returns log norm constant + log weight (base e) for
   *   @p compIdx-th component of @p gmmIdx-th GMM.
   *   GMM's and components are numbered starting from 0.
   **/
  double get_component_norm(unsigned gmmIdx, unsigned compIdx) const {
    assert(m_normsUpToDate && (gmmIdx < m_gmmMap.size()) &&
           (compIdx < get_component_count(gmmIdx)));
    return m_logNorms[get_min_comp_index(gmmIdx) + compIdx];
  }

 private:
  /** For each GMM, index of first component in m_compMap, etc. **/
  vector<unsigned> m_gmmMap;

  /** For each component of each GMM, its index in m_gaussParams. **/
  vector<unsigned> m_compMap;

  /** For each component of each GMM, its mixture weight. **/
  vector<double> m_compWeights;

  /** For each Gaussian, alternating mean + var for each dim. **/
  matrix<double> m_gaussParams;

  /** Whether m_logNorms is up to date. **/
  mutable bool m_normsUpToDate;

  /** For each component of each GMM, log norm constant + log weight. **/
  mutable vector<double> m_logNorms;
};

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   GMM count class.
 *
 *   Holds a posterior count for a GMM at a frame.  Includes the
 *   GMM index, the frame, and the posterior count.  This is
 *   used to facilitate Forward-Backward and Viterbi EM training.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class GmmCount {
 public:
  /** Ctor; initializes fields to default values. **/
  GmmCount() : m_gmmIdx(0), m_frmIdx(0), m_count(0.0) {}

  /** Ctor; explicitly initializes all fields. **/
  GmmCount(unsigned gmmIdx, unsigned frmIdx, double count)
      : m_gmmIdx(gmmIdx), m_frmIdx(frmIdx), m_count(count) {}

  /** Sets all values in object. **/
  void assign(unsigned gmmIdx, unsigned frmIdx, double count) {
    m_gmmIdx = gmmIdx;
    m_frmIdx = frmIdx;
    m_count = count;
  }

  /** Returns the associated GMM index. **/
  unsigned get_gmm_index() const { return m_gmmIdx; }

  /** Returns the associated frame index. **/
  unsigned get_frame_index() const { return m_frmIdx; }

  /** Returns the posterior count. **/
  double get_count() const { return m_count; }

 private:
  /** The index of the GMM. **/
  unsigned m_gmmIdx;

  /** Which frame the count occurred at. **/
  unsigned m_frmIdx;

  /** The posterior count. **/
  float m_count;
};

#ifndef SWIG

/** Orders GmmCount objects first by frame, then GMM index, then
 *   by decreasing count.
 **/
inline bool operator<(const GmmCount& cnt1, const GmmCount& cnt2) {
  if (cnt1.get_frame_index() != cnt2.get_frame_index())
    return cnt1.get_frame_index() < cnt2.get_frame_index();
  if (cnt1.get_gmm_index() != cnt2.get_gmm_index())
    return cnt1.get_gmm_index() < cnt2.get_gmm_index();
  return cnt1.get_count() > cnt2.get_count();
}

#endif

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Class holding symbol table for a graph/FSM.
 *
 *   In graphs/FSM's, word labels are stored internally as integers.
 *   This object holds the mapping from word spellings to their
 *   integer representations, and vice versa.  Word indices are
 *   constrained to be nonnegative, though they need not be consecutive.
 *   By convention, the word
 *   index corresponding to epsilon (i.e., the representation of
 *   the empty string) is 0.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class SymbolTable {
 public:
  /** Ctor; loads from file @p fileName if argument present. **/
  SymbolTable(const string& fileName = string()) {
    if (!fileName.empty()) read(fileName);
  }

  /** Reads symbols from file @p fileName. **/
  void read(const string& fileName);

  /** Clears object. **/
  void clear();

  /** Returns number of symbols in table. **/
  unsigned size() const { return m_strToIdxMap.size(); }

  /** Returns whether symbol table is empty. **/
  bool empty() const { return m_strToIdxMap.empty(); }

  /** Maps from a string to its index.
   *   If not in table, returns -1.
   **/
  int get_index(const string& theStr) const {
    map<string, unsigned>::const_iterator lookup = m_strToIdxMap.find(theStr);
    return (lookup != m_strToIdxMap.end()) ? (int)lookup->second : -1;
  }

  /** Maps from an index to its string.
   *   If not in table, returns empty string.
   **/
  string get_str(unsigned theIdx) const {
    map<unsigned, string>::const_iterator lookup = m_idxToStrMap.find(theIdx);
    return (lookup != m_idxToStrMap.end()) ? lookup->second : string();
  }

 private:
  /** Map from strings to integer indices. **/
  map<string, unsigned> m_strToIdxMap;

  /** Map from integer indices to strings. **/
  map<unsigned, string> m_idxToStrMap;
};

/** For converting a vector of strings @p wordList to a vector of
 *   ints @p wordIdxList using a SymbolTable, for n-gram model processing.
 *   Words not in the symbol table are converted
 *   to the value @p unkIdx.  The beginning of the output sequence
 *   is padded with @p n - 1 @p bosIdx values, and a single @p eosIdx
 *   value is added to the end.
 **/
void convert_words_to_indices(const vector<string>& wordList,
                              vector<int>& wordIdxList,
                              const SymbolTable& symTable, int n, int bosIdx,
                              int eosIdx, int unkIdx);

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Arc class.
 *
 *   Holds a single arc in a graph/FSM.  With each graph, there is
 *   an implicitly associated GmmSet and an explicitly associated
 *   SymbolTable (see Graph::get_word_sym_table()).
 *   An arc holds a destination state;
 *   an optional GMM index (corresponding to a GMM in the GmmSet);
 *   an optional word index (corresponding to a word in the SymbolTable);
 *   and a log prob (base e).
 *   Source state information is not present; if you have the arc ID
 *   (see Graph::get_first_arc_id(), Graph::get_arc()), you can look this
 *   up using Graph::get_src_state().
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class Arc {
 public:
  /** Ctor; initializes fields to default values. **/
  Arc() : m_dst(0), m_gmmIdx(-1), m_wordIdx(0), m_logProb(0.0) {}

  /** Ctor; explicitly initializes all fields.
   *   @see #assign().
   **/
  Arc(unsigned dst, int gmmIdx, int wordIdx, double logProb)
      : m_dst(dst), m_gmmIdx(gmmIdx), m_wordIdx(wordIdx), m_logProb(logProb) {}

  /** Sets all values in arc.
   *   The argument @p dst should be the destination state;
   *   @p gmmIdx should be the index of the associated GMM (or -1
   *   if not present); @p wordIdx should be the index of the
   *   associated word (or -1 if not present); and @p logProb
   *   should be the log prob base e of the arc.
   **/
  void assign(unsigned dst, int gmmIdx, int wordIdx, double logProb) {
    m_dst = dst;
    m_gmmIdx = gmmIdx;
    m_wordIdx = wordIdx;
    m_logProb = logProb;
  }

  /** Returns dest state index.
   *   To find src state, see Graph::get_src_state().
   **/
  unsigned get_dst_state() const { return m_dst; }

  /** Returns assoc GMM index, or -1 if not present. **/
  int get_gmm() const { return m_gmmIdx; }

  /** Returns assoc word index, or 0 if not present/epsilon.
   *   To find the corresponding word spelling, see
   *   Graph::get_word_sym_table().
   **/
  unsigned get_word() const { return m_wordIdx; }

  /** Returns assoc log prob base e. **/
  double get_log_prob() const { return m_logProb; }

 private:
  /** Destination state. **/
  unsigned m_dst;

  /** GMM index, or -1 if not present. **/
  int m_gmmIdx;

  /** Word index, or 0 if not present/epsilon. **/
  unsigned m_wordIdx;

  /** Log prob base e. **/
  float m_logProb;
};

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Graph/FSM class.
 *
 *   This object holds a graph as needed for training or decoding.
 *   A graph has a set of states numbered starting from 0
 *   (see #get_state_count()), one of
 *   which is a start state (see #get_start_state()),
 *   and many of which may be final states (see #is_final_state(),
 *   #get_final_state_list()).  Final states have an associated final
 *   log prob (see #get_final_log_prob()).
 *
 *   For each state, you can access the list of outgoing arcs using
 *   #get_arc_count(), #get_first_arc_id(), and #get_arc().
 *   Each arc has an associated ID, which is used for iterating
 *   through arcs.  Here is an example of how to iterate through
 *   the outgoing arcs of state @p stateIdx of graph @p graph.
 *
 *   @code
 *   //  Get number of outgoing arcs.
 *   int arcCnt = graph.get_arc_count(stateIdx);
 *   //  Get arc ID of first outgoing arc.
 *   int arcId = graph.get_first_arc_id(stateIdx);
 *   for (int arcIdx = 0; arcIdx < arcCnt; ++arcIdx)
 *       {
 *       Arc arc;
 *       //  Place arc with ID "arcId" in "arc"; set "arcId"
 *       //  to arc ID of the next outgoing arc.
 *       arcId = graph.get_arc(arcId, arc);
 *
 *       //  You can now access elements of the Arc "arc", e.g.,
 *       int dstState = arc.get_dst_state();
 *       ...
 *       }
 *   @endcode
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class Graph {
 public:
  /** Ctor; loads from file @p fileName if argument present,
   *   and loads sym table from file @p symFile if argument present.
   **/
  Graph(const string& fileName = string(), const string& symFile = string());

  /** Reads graph from file @p fileName.
   *   Reads symbol table from file @p symFile if not empty string.
   **/
  void read(const string& fileName, const string& symFile = string());

  /** Reads graph from stream @p inStrm.
   *   If argument @p name is provided, checks that name associated with
   *   graph matches.  Returns name given in graph header,
   *   or empty string if none provided.
   **/
  string read(istream& inStrm, const string& name = string());

  /** Writes graph to file @p fileName. **/
  void write(const string& fileName) const;

  /** Writes graph to stream @p outStrm.
   *   If the argument @p name is provided, this name will be written
   *   in the graph header.
   **/
  void write(ostream& outStrm, const string& name = string()) const;

  /** Reads word symbol table from file @p symFile.
   *   Pass in the empty string to load an empty symbol table.
   **/
  void read_word_sym_table(const string& symFile);

  /** Clears object except for symbol table; i.e., delete all states
   *   and arcs.
   **/
  void clear();

  /** Returns whether there are no states. **/
  bool empty() const { return m_stateMap.empty(); }

  /** Returns a reference to the word symbol table. **/
  const SymbolTable& get_word_sym_table() const { return *m_symTable.get(); }

  /** Returns one above highest GMM index in graph. **/
  unsigned get_gmm_count() const;

  /** Returns total number of states. **/
  unsigned get_state_count() const { return m_stateMap.size(); }

  /** Returns index of start state, or -1 if unset. **/
  int get_start_state() const { return m_start; }

  /** Returns number of outgoing arcs for state @p stateIdx.
   *   States are numbered from 0.
   **/
  unsigned get_arc_count(unsigned stateIdx) const {
    return get_max_arc_index(stateIdx) - get_min_arc_index(stateIdx);
  }

  /** Returns arc ID of first outgoing arc of state @p stateIdx.
   *   States are numbered from 0.
   **/
  unsigned get_first_arc_id(unsigned stateIdx) const {
    assert(stateIdx < m_stateMap.size());
    return get_min_arc_index(stateIdx);
  }

  /** Places arc with ID @p arcId in @p arc.
   *   Returns ID of next outgoing arc of same state.
   **/
  unsigned get_arc(unsigned arcId, Arc& arc) const {
    assert(arcId < m_arcList.size());
    arc = m_arcList[arcId];
    return arcId + 1;
  }

  /** Returns src state of an arc given its ID. **/
  unsigned get_src_state(unsigned arcId) const;

  /** Returns wheter state @p stateIdx is final state.
   *   States are numbered from 0.
   **/
  bool is_final_state(unsigned stateIdx) const {
    return m_finalLogProbs.find(stateIdx) != m_finalLogProbs.end();
  }

  /** Returns final log prob (base e) of state @p stateIdx or
   *   g_zeroLogProb if not final.
   *   States are numbered from 0.
   **/
  double get_final_log_prob(unsigned stateIdx) const {
    map<unsigned, float>::const_iterator lookup =
        m_finalLogProbs.find(stateIdx);
    return (lookup != m_finalLogProbs.end()) ? lookup->second : g_zeroLogProb;
  }

  /** Returns number of final states; places list in @p stateList. **/
  unsigned get_final_state_list(vector<int>& stateList) const {
    stateList.clear();
    for (map<unsigned, float>::const_iterator elemPtr = m_finalLogProbs.begin();
         elemPtr != m_finalLogProbs.end(); ++elemPtr)
      stateList.push_back(elemPtr->first);
    sort(stateList.begin(), stateList.end());
    return stateList.size();
  }

 private:
  /** Returns index of first arc for state @p stateIdx. **/
  unsigned get_min_arc_index(unsigned stateIdx) const {
    assert(stateIdx < m_stateMap.size());
    return m_stateMap[stateIdx];
  }

  /** Returns one past index of last arc for state @p stateIdx. **/
  unsigned get_max_arc_index(unsigned stateIdx) const {
    assert(stateIdx < m_stateMap.size());
    return (stateIdx + 1 < m_stateMap.size()) ? m_stateMap[stateIdx + 1]
                                              : m_arcList.size();
  }

 private:
  /** Map from words to integer indices. **/
  shared_ptr<SymbolTable> m_symTable;

  /** Index of start state. **/
  int m_start;

  /** Map from final states to their final log probs. **/
  map<unsigned, float> m_finalLogProbs;

  /** For each state, index of first arc in m_arcList.
   *   Assumes that arcs for each state are contiguous.
   **/
  vector<unsigned> m_stateMap;

  /** List of arcs in graph, grouped by source state. **/
  vector<Arc> m_arcList;
};

/** * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * **
 *   Class for storing counts for a set of n-grams.
 *
 *   N-grams are represented as vectors of integers.  The main access
 *   methods are #incr_count(), #set_count(), and #get_count().
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
class NGramCounter {
 public:
  /** Ctor; initializes object to be empty. **/
  NGramCounter() {}

  /** Writes all counts to stream @p outStrm in a text format.
   *   Uses @p symTable to map integer indices to strings, if present.
   **/
  void write(ostream& outStrm,
             const SymbolTable& symTable = SymbolTable()) const;

  /** Clears object; deletes all n-grams in table. **/
  void clear() { m_countMap.clear(); }

  /** Returns number of n-grams in table. **/
  unsigned size() const { return m_countMap.size(); }

  /** Returns whether object is empty. **/
  bool empty() const { return m_countMap.empty(); }

  /** Increments count of an n-gram; returns new count. **/
  unsigned incr_count(const vector<int>& ngram) { return ++m_countMap[ngram]; }

  /** Sets count of an n-gram to @p val.
   *   If @p val is 0, n-gram is removed from table.
   **/
  void set_count(const vector<int>& ngram, unsigned val) {
    if (val != 0)
      m_countMap[ngram] = val;
    else
      m_countMap.erase(ngram);
  }

  /** Returns count of an n-gram, or 0 if not present. **/
  unsigned get_count(const vector<int>& ngram) const {
    map<vector<int>, int>::const_iterator iter = m_countMap.find(ngram);
    return (iter != m_countMap.end()) ? iter->second : 0;
  }

 private:
  /** Map from n-grams to counts. **/
  map<vector<int>, int> m_countMap;
};

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#endif
