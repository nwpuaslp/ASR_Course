
//  $Id: util.C,v 1.30 2009/09/28 15:50:47 stanchen Exp $

#include <boost/lexical_cast.hpp>
#include <utility>
#include "util.H"

using boost::bad_lexical_cast;
using boost::lexical_cast;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

double add_log_probs(const vector<double>& logProbList) {
  const double lprSmall = -7.0;  // e^(-7) ~= 0.001

  const double* plprBegin = &*logProbList.begin();
  const double* plprEnd = &*logProbList.end();
  assert(plprEnd > plprBegin);
  if (plprEnd == plprBegin + 1) return *plprBegin;
  double lprMax = g_zeroLogProb;
  for (const double* plprCur = plprBegin; plprCur != plprEnd; ++plprCur) {
    if (*plprCur > lprMax) lprMax = *plprCur;
  }
  double prSum = 0.0;
  for (const double* plprCur = plprBegin; plprCur != plprEnd; ++plprCur) {
    if (*plprCur == lprMax)
      prSum += 1.0;
    else {
      double lprNorm = *plprCur - lprMax;
      if (lprNorm > lprSmall) prSum += exp(lprNorm);
    }
  }
  assert(prSum >= 1.0);
  return (prSum == 1.0) ? lprMax : lprMax + log(prSum);
}

static void full_fft(int cflIn, double* rgfl, bool fInvert) {
  //  Make 1-based.
  --rgfl;

  int ii, jj, nn, limit, m, j, inc, i;
  double wx, wr, wpr, wpi, wi, theta;
  double xre, xri, x;

  nn = cflIn / 2;
  j = 1;
  for (ii = 1; ii <= nn; ii++) {
    i = 2 * ii - 1;
    if (j > i) {
      xre = rgfl[j];
      xri = rgfl[j + 1];
      rgfl[j] = rgfl[i];
      rgfl[j + 1] = rgfl[i + 1];
      rgfl[i] = xre;
      rgfl[i + 1] = xri;
    }
    m = cflIn / 2;
    while (m >= 2 && j > m) {
      j -= m;
      m /= 2;
    }
    j += m;
  };
  limit = 2;
  while (limit < cflIn) {
    inc = 2 * limit;
    theta = 2 * M_PI / limit;
    if (fInvert) theta = -theta;
    x = sin(0.5 * theta);
    wpr = -2.0 * x * x;
    wpi = sin(theta);
    wr = 1.0;
    wi = 0.0;
    for (ii = 1; ii <= limit / 2; ii++) {
      m = 2 * ii - 1;
      for (jj = 0; jj <= (cflIn - m) / inc; jj++) {
        i = m + jj * inc;
        j = i + limit;
        xre = wr * rgfl[j] - wi * rgfl[j + 1];
        xri = wr * rgfl[j + 1] + wi * rgfl[j];
        rgfl[j] = rgfl[i] - xre;
        rgfl[j + 1] = rgfl[i + 1] - xri;
        rgfl[i] = rgfl[i] + xre;
        rgfl[i + 1] = rgfl[i + 1] + xri;
      }
      wx = wr;
      wr = wr * wpr - wi * wpi + wr;
      wi = wi * wpr + wx * wpi + wi;
    }
    limit = inc;
  }
  if (fInvert)
    for (i = 1; i <= cflIn; i++) rgfl[i] = rgfl[i] / nn;
}

void real_fft(vector<double>& vals) {
  int cflIn = vals.size();
  double* rgfl = &*vals.begin();

  int n, n2, i, i1, i2, i3, i4;
  double xr1, xi1, xr2, xi2, wrs, wis;
  double yr, yi, yr2, yi2, yr0, theta, x;

  n = cflIn / 2;
  n2 = n / 2;
  theta = M_PI / n;
  full_fft(cflIn, rgfl, false);

  //  Make 1-based.
  --rgfl;

  x = sin(0.5 * theta);
  yr2 = -2.0 * x * x;
  yi2 = sin(theta);
  yr = 1.0 + yr2;
  yi = yi2;
  for (i = 2; i <= n2; i++) {
    i1 = i + i - 1;
    i2 = i1 + 1;
    i3 = n + n + 3 - i2;
    i4 = i3 + 1;
    wrs = yr;
    wis = yi;
    xr1 = (rgfl[i1] + rgfl[i3]) / 2.0;
    xi1 = (rgfl[i2] - rgfl[i4]) / 2.0;
    xr2 = (rgfl[i2] + rgfl[i4]) / 2.0;
    xi2 = (rgfl[i3] - rgfl[i1]) / 2.0;
    rgfl[i1] = xr1 + wrs * xr2 - wis * xi2;
    rgfl[i2] = xi1 + wrs * xi2 + wis * xr2;
    rgfl[i3] = xr1 - wrs * xr2 + wis * xi2;
    rgfl[i4] = -xi1 + wrs * xi2 + wis * xr2;
    yr0 = yr;
    yr = yr * yr2 - yi * yi2 + yr;
    yi = yi * yr2 + yr0 * yi2 + yi;
  }
  xr1 = rgfl[1];
  rgfl[1] = xr1 + rgfl[2];
  rgfl[2] = 0.0;
}

void copy_matrix_row_to_vector(const matrix<double>& mat, unsigned rowIdx,
                               vector<double>& vec) {
  assert(rowIdx < mat.size1());
  vec.resize(mat.size2());
  if (mat.size2())
    copy(&mat(rowIdx, 0), &mat(rowIdx, 0) + mat.size2(), vec.begin());
}

void copy_vector_to_matrix_row(const vector<double>& vec, matrix<double>& mat,
                               unsigned rowIdx) {
  assert((rowIdx < mat.size1()) && (vec.size() == mat.size2()));
  if (vec.size()) copy(vec.begin(), vec.end(), &mat(rowIdx, 0));
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void process_cmd_line(const vector<string>& argList,
                      map<string, string>& params) {
  for (int argIdx = 0; argIdx < (int)argList.size();) {
    string flagStr = argList[argIdx++];
    if ((flagStr.length() < 3) || (flagStr.substr(0, 2) != "--"))
      throw runtime_error("Invalid flag name: " + flagStr);
    if (argIdx >= (int)argList.size())
      throw runtime_error("Unexpected EOL after: " + flagStr);
    string valStr = argList[argIdx++];
    params[flagStr.substr(2, string::npos)] = valStr;
  }
}

void process_cmd_line(const string& argStr, map<string, string>& params) {
  vector<string> argList;
  split_string(argStr, argList);
  process_cmd_line(argList, params);
}

void process_cmd_line(const char** argv, map<string, string>& params) {
  if (!*argv) throw runtime_error("Invalid argument list.");
  vector<string> argList;
  for (const char** argPtr = argv + 1; *argPtr; ++argPtr)
    argList.push_back(*argPtr);
  process_cmd_line(argList, params);
}

bool get_bool_param(const map<string, string>& params, const string& name,
                    bool defaultVal) {
  map<string, string>::const_iterator lookup = params.find(name);
  if (lookup == params.end()) return defaultVal;
  string val = lookup->second;
  if ((val == "1") or (val == "true")) return true;
  if ((val == "0") or (val == "false")) return false;
  throw runtime_error("Invalid value for bool param: " + name);
}

int get_int_param(const map<string, string>& params, const string& name,
                  int defaultVal) {
  map<string, string>::const_iterator lookup = params.find(name);
  if (lookup == params.end()) return defaultVal;
  try {
    return lexical_cast<int>(lookup->second);
  } catch (bad_lexical_cast&) {
    throw runtime_error("Invalid value for int param: " + name);
  }
}

double get_float_param(const map<string, string>& params, const string& name,
                       double defaultVal) {
  map<string, string>::const_iterator lookup = params.find(name);
  if (lookup == params.end()) return defaultVal;
  try {
    return lexical_cast<double>(lookup->second);
  } catch (bad_lexical_cast&) {
    throw runtime_error("Invalid value for float param: " + name);
  }
}

string get_string_param(const map<string, string>& params, const string& name,
                        const string& defaultVal) {
  map<string, string>::const_iterator lookup = params.find(name);
  if (lookup == params.end()) return defaultVal;
  return lookup->second;
}

string get_required_string_param(const map<string, string>& params,
                                 const string& name) {
  map<string, string>::const_iterator lookup = params.find(name);
  if (lookup == params.end())
    throw runtime_error("Required string parameter missing: " + name);
  return lookup->second;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void split_string(const string& inStr, vector<string>& outList) {
  outList.clear();
  string::size_type curPos = inStr.find_first_not_of(" \n\t", 0);
  while (curPos != string::npos) {
    string::size_type endPos = inStr.find_first_of(" \n\t", curPos);
    if (endPos == string::npos) {
      outList.push_back(inStr.substr(curPos, string::npos));
      break;
    }
    outList.push_back(inStr.substr(curPos, endPos - curPos));
    curPos = inStr.find_first_not_of(" \n\t", endPos);
  }
}

void read_string_list(const string& fileName, vector<string>& strList) {
  strList.clear();
  ifstream inStrm(fileName.c_str());
  string lineStr;
  vector<string> fieldList;
  while (getline(inStrm, lineStr)) {
    split_string(lineStr, fieldList);
    if (!fieldList.size()) continue;
    if (fieldList.size() > 1)
      throw runtime_error(
          str(format("Invalid line (%s): %s") % fileName % lineStr));
    strList.push_back(fieldList[0]);
  }
}

template <typename T>
static string read_matrix(istream& inStrm, matrix<T>& mat, const string& name) {
  //  Read header.
  int rowCntCheck = -1;
  int colCnt = -1;
  string retStr;
  string lineStr;
  vector<string> fieldList;
  while (true) {
    int peekChar = inStrm.peek();
    if ((peekChar != '%') && (peekChar != '#')) break;
    getline(inStrm, lineStr);
    split_string(lineStr, fieldList);
    if ((fieldList.size() == 3) &&
        ((fieldList[0] == "%") || (fieldList[0] == "#"))) {
      if (fieldList[1] == "rows:")
        rowCntCheck = lexical_cast<int>(fieldList[2]);
      else if (fieldList[1] == "columns:")
        colCnt = lexical_cast<int>(fieldList[2]);
      else if (fieldList[1] == "name:") {
        if (!name.empty() && (fieldList[2] != name))
          throw runtime_error(str(format("Unexpected matrix name: %s/%s") %
                                  name % fieldList[2]));
        if (!retStr.empty())
          throw runtime_error(str(format("Matrix has two names: %s/%s") %
                                  retStr % fieldList[2]));
        retStr = fieldList[2];
      }
    }
  }

  //  Read body; read into vector first, because dynamic resizing
  //  of matrix is slow.
  vector<T> buf;
  int rowCntRead = 0;
  while (true) {
    int peekChar = inStrm.peek();
    if ((peekChar == '%') || (peekChar == '#') || (peekChar == EOF)) break;
    getline(inStrm, lineStr);
    split_string(lineStr, fieldList);
    if (fieldList.size() == 0)
      ;
    else {
      if (colCnt == -1) colCnt = fieldList.size();
      if ((int)fieldList.size() != colCnt)
        throw runtime_error("Invalid num fields: " + lineStr);
      try {
        for (unsigned colIdx = 0; (int)colIdx < colCnt; ++colIdx)
          buf.push_back(lexical_cast<T>(fieldList[colIdx]));
      } catch (bad_lexical_cast&) {
        throw runtime_error("Invalid value for matrix element.");
      }
    }
    ++rowCntRead;
  }
  if ((rowCntCheck != -1) && (rowCntRead != rowCntCheck))
    throw runtime_error("Mismatch in number of rows in matrix.");
  mat.resize(rowCntRead, (colCnt != -1) ? colCnt : 0, false);
  copy(buf.begin(), buf.end(), mat.data().begin());
  return retStr;
}

string read_float_matrix(istream& inStrm, matrix<double>& mat,
                         const string& name) {
  return read_matrix(inStrm, mat, name);
}

string read_float_vector(istream& inStrm, vector<double>& vec,
                         const string& name) {
  matrix<double> mat;
  string retStr = read_matrix(inStrm, mat, name);
  if (mat.size2() != 1)
    throw runtime_error("Invalid vector file: not single column.");
  vec.clear();
  vec.insert(vec.end(), mat.data().begin(), mat.data().end());
  return retStr;
}

string read_int_matrix(istream& inStrm, matrix<int>& mat, const string& name) {
  return read_matrix(inStrm, mat, name);
}

string read_int_vector(istream& inStrm, vector<int>& vec, const string& name) {
  matrix<int> mat;
  string retStr = read_matrix(inStrm, mat, name);
  if (mat.size2() != 1)
    throw runtime_error("Invalid vector file: not single column.");
  vec.clear();
  vec.insert(vec.end(), mat.data().begin(), mat.data().end());
  return retStr;
}

string read_unsigned_vector(istream& inStrm, vector<unsigned>& vec,
                            const string& name) {
  matrix<int> mat;
  string retStr = read_matrix(inStrm, mat, name);
  if (mat.size2() != 1)
    throw runtime_error("Invalid vector file: not single column.");
  vec.clear();
  vec.insert(vec.end(), mat.data().begin(), mat.data().end());
  return retStr;
}

void read_float_matrix(const string& fileName, matrix<double>& mat) {
  ifstream inStrm(fileName.c_str());
  read_float_matrix(inStrm, mat);
}

void read_float_vector(const string& fileName, vector<double>& vec) {
  ifstream inStrm(fileName.c_str());
  read_float_vector(inStrm, vec);
}

void read_int_matrix(const string& fileName, matrix<int>& mat) {
  ifstream inStrm(fileName.c_str());
  read_int_matrix(inStrm, mat);
}

void read_int_vector(const string& fileName, vector<int>& vec) {
  ifstream inStrm(fileName.c_str());
  read_int_vector(inStrm, vec);
}

void write_float_matrix(ostream& outStrm, const matrix<double>& mat,
                        const string& name) {
  if (!name.empty()) {
    outStrm << "% name: " << name << "\n";
    outStrm << "% type: matrix\n";
    outStrm << "% rows: " << mat.size1() << "\n";
    outStrm << "% columns: " << mat.size2() << "\n";
  }
  for (unsigned rowIdx = 0; rowIdx < mat.size1(); ++rowIdx) {
    for (unsigned colIdx = 0; colIdx < mat.size2(); ++colIdx)
      outStrm << " " << format("%g") % mat(rowIdx, colIdx);
    outStrm << "\n";
  }
}

void write_float_vector(ostream& outStrm, const vector<double>& vec,
                        const string& name) {
  matrix<double> mat(vec.size(), 1);
  copy(vec.begin(), vec.end(), mat.begin1());
  write_float_matrix(outStrm, mat, name);
}

void write_int_matrix(ostream& outStrm, const matrix<int>& mat,
                      const string& name) {
  if (!name.empty()) {
    outStrm << "% name: " << name << "\n";
    outStrm << "% type: matrix\n";
    outStrm << "% rows: " << mat.size1() << "\n";
    outStrm << "% columns: " << mat.size2() << "\n";
  }
  for (unsigned rowIdx = 0; rowIdx < mat.size1(); ++rowIdx) {
    for (unsigned colIdx = 0; colIdx < mat.size2(); ++colIdx)
      outStrm << " " << mat(rowIdx, colIdx);
    outStrm << "\n";
  }
}

void write_int_vector(ostream& outStrm, const vector<int>& vec,
                      const string& name) {
  matrix<int> mat(vec.size(), 1);
  copy(vec.begin(), vec.end(), mat.begin1());
  write_int_matrix(outStrm, mat, name);
}

void write_unsigned_vector(ostream& outStrm, const vector<unsigned>& vec,
                           const string& name) {
  matrix<int> mat(vec.size(), 1);
  copy(vec.begin(), vec.end(), mat.begin1());
  write_int_matrix(outStrm, mat, name);
}

void write_float_matrix(const string& fileName, const matrix<double>& mat) {
  ofstream outStrm(fileName.c_str());
  write_float_matrix(outStrm, mat);
}

void write_float_vector(const string& fileName, const vector<double>& vec) {
  ofstream outStrm(fileName.c_str());
  write_float_vector(outStrm, vec);
}

void write_int_matrix(const string& fileName, const matrix<int>& mat) {
  ofstream outStrm(fileName.c_str());
  write_int_matrix(outStrm, mat);
}

void write_int_vector(const string& fileName, const vector<int>& vec) {
  ofstream outStrm(fileName.c_str());
  write_int_vector(outStrm, vec);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

GmmSet::GmmSet(const string& fileName) : m_normsUpToDate(false) {
  if (!fileName.empty()) read(fileName);
}

void GmmSet::read(const string& fileName) {
  clear();
  ifstream inStrm(fileName.c_str());
  read_unsigned_vector(inStrm, m_gmmMap, "gmmMap");
  read_unsigned_vector(inStrm, m_compMap, "compMap");
  read_float_vector(inStrm, m_compWeights, "compWeights");
  read_float_matrix(inStrm, m_gaussParams, "gaussParams");
  if (!m_gmmMap.size()) throw runtime_error("Empty GMM map.");
  int gmmCnt = m_gmmMap.size();
  for (int gmmIdx = 1; gmmIdx < gmmCnt; ++gmmIdx) {
    if (m_gmmMap[gmmIdx] <= m_gmmMap[gmmIdx - 1])
      throw runtime_error("Invalid GMM map.");
  }
  if (m_gmmMap.back() >= m_compMap.size())
    throw runtime_error("Inconsistent GMM map and component map.");
  int compCnt = m_compMap.size();
  for (int compIdx = 0; compIdx < compCnt; ++compIdx) {
    if (m_compMap[compIdx] >= m_gaussParams.size1())
      throw runtime_error("Out-of-bounds component map entry.");
  }
  if (m_compWeights.size() != m_compMap.size())
    throw runtime_error(
        "Mismatch in size between component map and "
        "component weights.");
  if (m_gaussParams.size2() & 1)
    throw runtime_error("GMM params have odd number of columns.");
  assert(!m_normsUpToDate);
}

void GmmSet::write(const string& fileName) const {
  ofstream outStrm(fileName.c_str());
  write_unsigned_vector(outStrm, m_gmmMap, "gmmMap");
  write_unsigned_vector(outStrm, m_compMap, "compMap");
  write_float_vector(outStrm, m_compWeights, "compWeights");
  write_float_matrix(outStrm, m_gaussParams, "gaussParams");
}

void GmmSet::init(const vector<int>& gmmCompCounts, unsigned dimCnt,
                  const vector<int>& compMap) {
  clear();
  int gmmCnt = gmmCompCounts.size();
  int compCnt = 0;
  m_gmmMap.reserve(gmmCnt);
  for (int gmmIdx = 0; gmmIdx < gmmCnt; ++gmmIdx) {
    if (gmmCompCounts[gmmIdx] <= 0)
      throw runtime_error("GMM w/ nonpositive number of components.");
    m_gmmMap.push_back(compCnt);
    compCnt += gmmCompCounts[gmmIdx];
  }
  assert((int)m_gmmMap.size() == gmmCnt);

  int gaussCnt = 0;
  if (!compMap.empty()) {
    if ((int)compMap.size() != compCnt)
      throw runtime_error(
          "Mismatch between GMM component counts and "
          "component map.");
    m_compMap.insert(m_compMap.end(), compMap.begin(), compMap.end());
    for (int compIdx = 0; compIdx < compCnt; ++compIdx) {
      int gaussIdx = m_compMap[compIdx];
      if (gaussIdx < 0) throw runtime_error("Negative component map entry.");
      if (gaussIdx + 1 > gaussCnt) gaussCnt = gaussIdx + 1;
    }
  } else {
    m_compMap.reserve(compCnt);
    for (int compIdx = 0; compIdx < compCnt; ++compIdx)
      m_compMap.push_back(compIdx);
    gaussCnt = compCnt;
  }
  assert((int)m_compMap.size() == compCnt);

  m_compWeights.reserve(compCnt);
  for (int gmmIdx = 0; gmmIdx < gmmCnt; ++gmmIdx)
    m_compWeights.insert(m_compWeights.end(), gmmCompCounts[gmmIdx],
                         1.0 / gmmCompCounts[gmmIdx]);
  assert((int)m_compWeights.size() == compCnt);

  m_gaussParams.resize(gaussCnt, dimCnt * 2);
  for (int gaussIdx = 0; gaussIdx < gaussCnt; ++gaussIdx) {
    for (int dimIdx = 0; dimIdx < (int)dimCnt; ++dimIdx) {
      set_gaussian_mean(gaussIdx, dimIdx, 0.0);
      set_gaussian_var(gaussIdx, dimIdx, 1.0);
    }
  }
  assert(!m_normsUpToDate);
}

void GmmSet::clear() {
  m_gmmMap.clear();
  m_compMap.clear();
  m_compWeights.clear();
  m_gaussParams.resize(0, 0);
  m_normsUpToDate = false;
  m_logNorms.clear();
}

void GmmSet::copy_gaussian(unsigned dstGaussIdx, const GmmSet& srcGmmSet,
                           unsigned srcGaussIdx) {
  assert(srcGmmSet.get_dim_count() == get_dim_count());
  copy(&srcGmmSet.m_gaussParams(srcGaussIdx, 0),
       &srcGmmSet.m_gaussParams(srcGaussIdx, 0) + m_gaussParams.size2(),
       &m_gaussParams(dstGaussIdx, 0));
  m_normsUpToDate = false;
}

void GmmSet::compute_norms() const {
  assert(!m_normsUpToDate);
  m_logNorms.clear();
  m_logNorms.reserve(m_compMap.size());
  int gmmCnt = get_gmm_count();
  int dimCnt = get_dim_count();
  for (int gmmIdx = 0; gmmIdx < gmmCnt; ++gmmIdx) {
    int compCnt = get_component_count(gmmIdx);
    for (int compIdx = 0; compIdx < compCnt; ++compIdx) {
      int gaussIdx = get_gaussian_index(gmmIdx, compIdx);
      double logNorm = dimCnt * log(2.0 * M_PI);
      for (int dimIdx = 0; dimIdx < dimCnt; ++dimIdx) {
        if (m_gaussParams(gaussIdx, 2 * dimIdx + 1) <= 0.0)
          throw runtime_error("Gaussian w/ nonpositive variance.");
        logNorm += log(m_gaussParams(gaussIdx, 2 * dimIdx + 1));
      }
      double wgt = get_component_weight(gmmIdx, compIdx);
      m_logNorms.push_back(-0.5 * logNorm +
                           ((wgt > 0.0) ? log(wgt) : g_zeroLogProb));
    }
  }
  assert(m_logNorms.size() == m_compMap.size());
  m_normsUpToDate = true;
}

void GmmSet::calc_gmm_probs(const matrix<double>& feats,
                            matrix<double>& logProbs) const {
  if (!m_normsUpToDate) compute_norms();
  int dimCnt = get_dim_count();
  if ((int)feats.size2() != dimCnt)
    throw runtime_error("Mismatch in dims of GMM's and features.");
  int gmmCnt = get_gmm_count();
  logProbs.resize(feats.size1(), gmmCnt);
  vector<double> logProbList;
  for (int gmmIdx = 0; gmmIdx < gmmCnt; ++gmmIdx) {
    int compCnt = get_component_count(gmmIdx);
    for (int frmIdx = 0; frmIdx < (int)feats.size1(); ++frmIdx) {
      logProbList.clear();
      for (int compIdx = 0; compIdx < compCnt; ++compIdx) {
        int gaussIdx = get_gaussian_index(gmmIdx, compIdx);
        double logProbSum = 0.0;
        for (int dimIdx = 0; dimIdx < dimCnt; ++dimIdx) {
          double featDiff =
              feats(frmIdx, dimIdx) - m_gaussParams(gaussIdx, 2 * dimIdx);
          logProbSum +=
              featDiff * featDiff / m_gaussParams(gaussIdx, 2 * dimIdx + 1);
        }
        logProbList.push_back(-0.5 * logProbSum +
                              get_component_norm(gmmIdx, compIdx));
      }
      logProbs(frmIdx, gmmIdx) = add_log_probs(logProbList);
    }
  }
}

double GmmSet::calc_component_probs(const vector<double>& feats,
                                    unsigned gmmIdx,
                                    vector<double>& logProbs) const {
  if (!m_normsUpToDate) compute_norms();
  logProbs.clear();
  int dimCnt = get_dim_count();
  if ((int)feats.size() != dimCnt)
    throw runtime_error("Mismatch in dims of GMM's and features.");
  int compCnt = get_component_count(gmmIdx);
  for (int compIdx = 0; compIdx < compCnt; ++compIdx) {
    int gaussIdx = get_gaussian_index(gmmIdx, compIdx);
    double logProbSum = 0.0;
    for (int dimIdx = 0; dimIdx < dimCnt; ++dimIdx) {
      double featDiff = feats[dimIdx] - m_gaussParams(gaussIdx, 2 * dimIdx);
      logProbSum +=
          featDiff * featDiff / m_gaussParams(gaussIdx, 2 * dimIdx + 1);
    }
    logProbs.push_back(-0.5 * logProbSum + get_component_norm(gmmIdx, compIdx));
  }
  return add_log_probs(logProbs);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

void SymbolTable::read(const string& fileName) {
  clear();
  ifstream inStrm(fileName.c_str());
  string lineStr;
  vector<string> fieldList;
  while (getline(inStrm, lineStr)) {
    split_string(lineStr, fieldList);
    if (!fieldList.size()) continue;
    if (fieldList.size() != 2)
      throw runtime_error(str(format("Invalid line in sym table (%s): %s") %
                              fileName % lineStr));
    int theIdx = -1;
    string theStr = fieldList[0];
    try {
      theIdx = lexical_cast<int>(fieldList[1]);
    } catch (bad_lexical_cast&) {
      throw runtime_error(str(format("Invalid line in sym table (%s): %s") %
                              fileName % lineStr));
    }
    if (theIdx < 0)
      throw runtime_error(str(format("Negative index in sym table (%s): %s") %
                              fileName % lineStr));
    if ((m_strToIdxMap.find(theStr) != m_strToIdxMap.end()) ||
        (m_idxToStrMap.find(theIdx) != m_idxToStrMap.end()))
      throw runtime_error(str(format("Duplicate entry in sym table (%s): %s") %
                              fileName % lineStr));
    m_strToIdxMap[theStr] = theIdx;
    m_idxToStrMap[theIdx] = theStr;
  }
}

void SymbolTable::clear() {
  m_strToIdxMap.clear();
  m_idxToStrMap.clear();
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

struct CompareArcs {
  bool operator()(const pair<int, Arc>& arc1,
                  const pair<int, Arc>& arc2) const {
    if (arc1.first != arc2.first) return (arc1.first < arc2.first);
    if (arc1.second.get_gmm() != arc2.second.get_gmm())
      return (arc1.second.get_gmm() < arc2.second.get_gmm());
    if (arc1.second.get_word() != arc2.second.get_word())
      return (arc1.second.get_word() < arc2.second.get_word());
    if (arc1.second.get_dst_state() != arc2.second.get_dst_state())
      return (arc1.second.get_dst_state() < arc2.second.get_dst_state());
    return false;
  }
};

Graph::Graph(const string& fileName, const string& symFile)
    : m_symTable(new SymbolTable), m_start(-1) {
  if (!symFile.empty()) read_word_sym_table(symFile);
  if (!fileName.empty()) read(fileName);
}

string Graph::read(istream& inStrm, const string& name) {
  clear();

  //  Read header.
  string retStr;
  string lineStr;
  vector<string> fieldList;
  while (true) {
    int peekChar = inStrm.peek();
    if (peekChar != '#') break;
    getline(inStrm, lineStr);
    split_string(lineStr, fieldList);
    if ((fieldList.size() == 3) && (fieldList[0] == "#") &&
        (fieldList[1] == "name:")) {
      if (!name.empty() && (fieldList[2] != name))
        throw runtime_error(
            str(format("Unexpected FSM name: %s/%s") % name % fieldList[2]));
      if (!retStr.empty())
        throw runtime_error(
            str(format("FSM has two names: %s/%s") % retStr % fieldList[2]));
      retStr = fieldList[2];
    }
  }

  //  Read body.
  int lastIdx = -1;
  vector<pair<int, Arc> > arcList;
  double logFactor = -log(10.0);
  while (true) {
    int peekChar = inStrm.peek();
    if ((peekChar == '#') || (peekChar == EOF)) break;
    getline(inStrm, lineStr);
    split_string(lineStr, fieldList);
    if (!fieldList.size()) continue;
    try {
      int srcIdx = lexical_cast<int>(fieldList[0]);
      if (srcIdx < 0)
        throw runtime_error("Negative state index in FSM: " + lineStr);
      //  Set start state.
      if (m_start == -1) m_start = srcIdx;
      if (srcIdx > lastIdx) lastIdx = srcIdx;

      if (fieldList.size() <= 2) {
        //  Handle final states.
        double logProb = (fieldList.size() > 1)
                             ? lexical_cast<double>(fieldList[1]) * logFactor
                             : 0.0;
        if (m_finalLogProbs.find(srcIdx) != m_finalLogProbs.end())
          throw runtime_error("Dup final state in FSM: " + lineStr);
        m_finalLogProbs[srcIdx] = logProb;
        continue;
      }
      //  Handle regular arcs.
      if ((fieldList.size() == 3) || (fieldList.size() > 5))
        throw runtime_error("Invalid num fields in FSM: " + lineStr);
      int dstIdx = lexical_cast<int>(fieldList[1]);
      if (dstIdx < 0)
        throw runtime_error("Negative state index in FSM: " + lineStr);
      if (dstIdx > lastIdx) lastIdx = dstIdx;

      int gmmIdx = -1;
      const string& gmmStr = fieldList[2];
      if ((gmmStr.length() >= 3) && (gmmStr.length() <= 9) &&
          (gmmStr[0] == '<') && (gmmStr[gmmStr.length() - 1] == '>') &&
          (string("epsilon").substr(0, gmmStr.length() - 2) ==
           gmmStr.substr(1, gmmStr.length() - 2)))
        ;
      else {
        gmmIdx = lexical_cast<int>(gmmStr);
        if (gmmIdx < 0)
          throw runtime_error("Negative GMM index in FSM: " + lineStr);
      }
      int wordIdx =
          !m_symTable->empty() ? m_symTable->get_index(fieldList[3]) : 0;
      if (wordIdx < 0) throw runtime_error("OOV word in FSM: " + lineStr);
      double logProb = (fieldList.size() > 4)
                           ? lexical_cast<double>(fieldList[4]) * logFactor
                           : 0.0;
      Arc arc(dstIdx, gmmIdx, wordIdx, logProb);
      arcList.push_back(make_pair(srcIdx, arc));
    } catch (bad_lexical_cast&) {
      throw runtime_error("Invalid type for field in FSM: " + lineStr);
    }
  }
  if (m_start < 0) throw runtime_error("Empty FSM.");

  //  Build sorted arc list; state mapping table.
  int stateCnt = lastIdx + 1;
  m_stateMap.reserve(stateCnt);
  m_arcList.reserve(arcList.size());
  sort(arcList.begin(), arcList.end(), CompareArcs());
  for (int arcIdx = 0; arcIdx < (int)arcList.size(); ++arcIdx) {
    m_arcList.push_back(arcList[arcIdx].second);
    int srcIdx = arcList[arcIdx].first;
    while ((int)m_stateMap.size() <= srcIdx) m_stateMap.push_back(arcIdx);
  }
  while ((int)m_stateMap.size() < stateCnt)
    m_stateMap.push_back(arcList.size());
  assert(((int)m_stateMap.size() == stateCnt) &&
         (m_arcList.size() == arcList.size()));

  //  Double-check mapping table is correct.
  for (int stateIdx = 0; stateIdx < stateCnt; ++stateIdx) {
    int minArcIdx = get_min_arc_index(stateIdx);
    int maxArcIdx = get_max_arc_index(stateIdx);
    for (int arcIdx = minArcIdx; arcIdx < maxArcIdx; ++arcIdx)
      assert(arcList[arcIdx].first == stateIdx);
  }
  return retStr;
}

void Graph::read(const string& fileName, const string& symFile) {
  if (!symFile.empty()) read_word_sym_table(symFile);
  ifstream inStrm(fileName.c_str());
  read(inStrm);
}

void Graph::write(const string& fileName) const {
  ofstream outStrm(fileName.c_str());
  write(outStrm);
}

void Graph::write(ostream& outStrm, const string& name) const {
  if (!name.empty()) outStrm << "# name: " << name << "\n";
  int stateCnt = get_state_count();
  int startIdx = get_start_state();
  assert((startIdx == -1) || ((startIdx >= 0) && (startIdx < stateCnt)));
  if ((startIdx < 0) && (stateCnt > 0))
    throw runtime_error("Writing non-empty FSM with no start state.");
  if ((startIdx >= 0) && (stateCnt > 1) && !get_arc_count(startIdx))
    throw runtime_error("Writing FSM with no outgoing arcs from start state.");
  const SymbolTable& symTable = get_word_sym_table();
  double logFactor = -log(10.0);
  for (int loopIdx = 0; loopIdx < stateCnt; ++loopIdx) {
    int stateIdx = (loopIdx == 0)
                       ? startIdx
                       : ((loopIdx <= startIdx) ? loopIdx - 1 : loopIdx);
    int arcCnt = get_arc_count(stateIdx);
    int arcId = get_first_arc_id(stateIdx);
    for (int arcIdx = 0; arcIdx < arcCnt; ++arcIdx) {
      Arc arc;
      arcId = get_arc(arcId, arc);
      outStrm << format("%d\t%d") % stateIdx % arc.get_dst_state();
      if (arc.get_gmm() == -1)
        outStrm << "\t<epsilon>";
      else
        outStrm << "\t" << arc.get_gmm();
      if (symTable.empty())
        outStrm << "\t" << arc.get_word();
      else {
        string wordStr = symTable.get_str(arc.get_word());
        if (wordStr.empty()) throw runtime_error("Invalid word index in FSM.");
        outStrm << "\t" << wordStr;
      }
      if (arc.get_log_prob() != 0.0)
        outStrm << format("\t%.4f") % (arc.get_log_prob() / logFactor);
      outStrm << "\n";
    }
  }

  vector<int> finalList;
  int finalCnt = get_final_state_list(finalList);
  for (int finalIdx = 0; finalIdx < finalCnt; ++finalIdx) {
    int stateIdx = finalList[finalIdx];
    double logProb = get_final_log_prob(stateIdx);
    if (logProb == 0.0)
      outStrm << stateIdx << "\n";
    else
      outStrm << format("%d\t%.4f") % stateIdx % (logProb / logFactor) << "\n";
  }
}

void Graph::read_word_sym_table(const string& symFile) {
  if (!symFile.empty())
    m_symTable.reset(new SymbolTable(symFile));
  else
    m_symTable.reset(new SymbolTable);
}

void Graph::clear() {
  m_start = -1;
  m_finalLogProbs.clear();
  m_stateMap.clear();
  m_arcList.clear();
}

unsigned Graph::get_gmm_count() const {
  int lastIdx = -1;
  for (int arcIdx = 0; arcIdx < (int)m_arcList.size(); ++arcIdx) {
    int gmmIdx = m_arcList[arcIdx].get_gmm();
    if (gmmIdx > lastIdx) lastIdx = gmmIdx;
  }
  return lastIdx + 1;
}

unsigned Graph::get_src_state(unsigned arcIdx) const {
  assert(arcIdx < m_arcList.size());
  unsigned stateIdx =
      (upper_bound(m_stateMap.begin(), m_stateMap.end(), arcIdx) -
       m_stateMap.begin()) -
      1;
  assert((stateIdx < m_stateMap.size()) &&
         (arcIdx >= get_min_arc_index(stateIdx)) &&
         (arcIdx < get_max_arc_index(stateIdx)));
  return stateIdx;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
