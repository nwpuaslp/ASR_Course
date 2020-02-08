
//  $Id: front_end.C,v 1.2 2016/01/23 03:15:23 stanchen Exp $

#include "front_end.H"

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
static double inverse_mel_scale(double mel_freq) {
  return 700.0 * (exp(mel_freq / 1127.0) - 1.0);
}

static double mel_scale(double freq) {
  return 1127.0 * log(1.0 + freq / 700.0);
}

/** Module for doing windowing. **/
void FrontEnd::do_window(const matrix<double>& in_feats,
                         matrix<double>& out_feats) const {
  //  Get parameters.
  //  Input samples per second.
  double sample_rate = get_float_param(params_, "window.sample_rate", 20000.0);
  //  Output frames per second.
  double frames_per_sec =
      get_float_param(params_, "window.frames_per_sec", 100.0);
  //  Width of each window, in seconds.
  double window_width = get_float_param(params_, "window.window_size", 0.025);
  //  Whether to do Hamming or rectangular windowing.
  bool do_Hamming = get_bool_param(params_, "window.hamming", true);

  //  Get number of input samples.
  int in_samp_cnt = in_feats.size1();
  if (in_feats.size2() != 1)
    throw runtime_error("Windowing expected vector input.");

  //  Input sampling period in seconds.
  double sample_period = 1.0 / sample_rate;
  //  Output frame period, in seconds.
  double frame_period = 1.0 / frames_per_sec;
  //  Number of samples per window.
  int samp_per_window = (int)(window_width / sample_period + 0.5);
  //  Number of samples to shift between each window.
  int samp_shift = (int)(frame_period / sample_period + 0.5);
  //  Number of output frames.
  int out_frame_cnt = (in_samp_cnt - samp_per_window) / samp_shift + 1;

  //  Allocate output matrix and fill with zeros.
  out_feats.resize(out_frame_cnt, samp_per_window);
  out_feats.clear();

  //  BEGIN_LAB
  //
  //  Input:
  //      "in_feats", a matrix containing a single column holding the
  //      input samples for an utterance.  Each row holds a single sample.
  //
  //      in_feats(0 .. (in_samp_cnt - 1), 0)
  //
  //  Output:
  //      "out_feats", which should contain the result of windowing.
  //
  //      out_feats(0 .. (out_frame_cnt - 1), 0 .. (samp_per_window - 1))
  //
  //      Each row corresponds to a frame, and should hold the
  //      windowed samples for that frame.
  //      It has already been allocated to be of the correct size.
  //      If the boolean "do_Hamming" is true, then a Hamming
  //      window should be applied; otherwise, a rectangular
  //      window should be used.
  //
  //  See "in_samp_cnt", "samp_per_window", "samp_shift", and "out_frame_cnt"
  //  above for quantities you may (or may not) need for this computation.
  //
  //  When accessing matrices such as "in_feats" and "out_feats",
  //  use a syntax like "in_feats(frm_idx, dim_idx)" to access elements;
  //  using square brackets as in normal C arrays won't work.

  // cout << format("samp_shift %d\n") % samp_shift;
  // cout << format("out_frame_cnt %d\n") % out_frame_cnt;
  // cout << format("samp_perWindw %d\n") % samp_per_window;
  // cout << format("in_samp_cnt %d\n") % in_samp_cnt;
  if (do_Hamming) {
    // Hamming windows
    double a = 2 * M_PI / (samp_per_window - 1);
    std::vector<double> hamming_window(samp_per_window, 0);
    for (size_t n = 0; n < hamming_window.size(); n++ ) {
      hamming_window[n] = 0.54 - 0.46 * cos(a * n);
    }
    for (int i = 0; i < out_frame_cnt; i++) {
      int offset = i * samp_shift;
      for (int j = 0; j < samp_per_window; j++) {
        out_feats(i, j) = in_feats(offset+j, 0) * hamming_window[j];
      }
    }
  } else {
    // Rectangular window
    for (int i = 0; i < out_frame_cnt; i++) {
      int offset = i * samp_shift;
      for (int j = 0; j < samp_per_window; j++) {
        out_feats(i, j) = in_feats(offset+j, 0);
      }
    }
  }
  //  END_LAB
}

/** Module for doing FFT. **/
void FrontEnd::do_fft(const matrix<double>& in_feats,
                      matrix<double>& out_feats) const {
  //  Make output dimension the smallest power of 2 at least as
  //  large as input dimension.
  int in_frame_cnt = in_feats.size1();
  int in_dim_cnt = in_feats.size2();
  int out_dim_cnt = 2;
  while (out_dim_cnt < in_dim_cnt) out_dim_cnt *= 2;

  //  Allocate output matrix and fill with zeros.
  out_feats.resize(in_frame_cnt, out_dim_cnt);
  out_feats.clear();

  //  Input:
  //      "in_feats", a matrix with each row holding the windowed
  //      values for that frame.
  //
  //      in_feats(0 .. (in_frame_cnt - 1), 0 .. (in_dim_cnt - 1))
  //
  //  Output:
  //      "out_feats", where an FFT should be applied to each
  //      row/frame of "in_feats".
  //
  //      out_feats(0 .. (in_frame_cnt - 1), 0 .. (out_dim_cnt - 1))
  //
  //      For a given row/frame "frm_idx", the real and imaginary
  //      parts of the FFT value for frequency i/(out_dim_cnt*T)
  //      where T is the sample period are held in
  //      out_feats(frm_idx, 2*i) and out_feats(frm_idx, 2*i+1),
  //      respectively.

  vector<double> fft_buf;
  for (int frm_idx = 0; frm_idx < in_frame_cnt; ++frm_idx) {
    copy_matrix_row_to_vector(in_feats, frm_idx, fft_buf);
    //  Pad window with zeros, if needed.
    fft_buf.resize(out_dim_cnt, 0.0);
    real_fft(fft_buf);
    copy_vector_to_matrix_row(fft_buf, out_feats, frm_idx);
  }
}

/** Module for mel binning. **/
// change to google name style
void FrontEnd::do_melbin(const matrix<double>& in_feats,
                         matrix<double>& out_feats) const {
  //  Number of mel bins to make.
  int num_bins = get_int_param(params_, "melbin.bins", 26);
  //  Whether to take log of output or not.
  bool do_log = get_bool_param(params_, "melbin.log", true);
  //  Input samples per second.
  double sample_rate = get_float_param(params_, "window.sample_rate", 20000.0);
  double sample_period = 1.0 / sample_rate;

  //  Retrieve number of frames and dimension of input feature vectors.
  int in_frame_cnt = in_feats.size1();
  int in_dim_cnt = in_feats.size2();
  int out_dim_cnt = num_bins;

  //  Allocate output matrix and fill with zeros.
  out_feats.resize(in_frame_cnt, out_dim_cnt);
  out_feats.clear();

  //  BEGIN_LAB
  //
  //  Input:
  //      "in_feats", holding the output of a real FFT.
  //
  //      in_feats(0 .. (in_frame_cnt - 1), 0 .. (in_dim_cnt - 1))
  //
  //  Output:
  //      "out_feats", which should contain the result of
  //      mel-binning.
  //
  //      out_feats(0 .. (in_frame_cnt - 1), 0 .. (out_dim_cnt - 1))
  //
  //      If the boolean "doLog" is true,
  //      then each value should be replaced with its natural
  //      logarithm, or 0 if its logarithm is negative.
  //      "out_feats" has been allocated to be of the correct size.
  //
  //  See "in_frame_cnt", "in_dim_cnt", "out_dim_cnt", and "sample_period"
  //  above for quantities you will need for this computation.

  // cout << format("sample_period %ls\n") % sample_period;
  // cout << format("input dim %d\n") % in_dim_cnt;

  // TODO: Optimize the compuatation order to avoid duplicate computation
  int fft_size = in_dim_cnt;
  int energy_size = fft_size / 2;
  double mel_low_freq = mel_scale(0);
  double mel_high_freq = mel_scale(sample_rate / 2);
  double mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins+1);
  std::vector<double> center(num_bins + 2);
  for (int bin = 0; bin < num_bins + 2; bin++) {
    double mel_freq = mel_low_freq + bin * mel_freq_delta;
    center[bin] = inverse_mel_scale(mel_freq) *
                  fft_size / sample_rate;
  }

  std::vector<std::vector<double> > mel_weights(num_bins);
  for (int bin = 0; bin < num_bins; bin++) {
    mel_weights[bin].resize(energy_size);
    for (int fft_bin = 0; fft_bin < energy_size; fft_bin++) {
      if (fft_bin >= center[bin] && fft_bin <= center[bin+2]) {
        if (fft_bin <= center[bin+1]) {
          mel_weights[bin][fft_bin] = (double)(fft_bin - center[bin]) /
                                      (center[bin+1] - center[bin]);
        } else {
          mel_weights[bin][fft_bin] = (double)(center[bin+2] - fft_bin) /
                                      (center[bin+2] - center[bin+1]);
        }
      } else {
        mel_weights[bin][fft_bin] = 0.0;
      }
    }
  }

  for (int frm_idx= 0; frm_idx < in_frame_cnt; frm_idx++) {
    std::vector<double> energy(energy_size);
    for (int i = 0; i < energy_size; i++) {
      double real = in_feats(frm_idx, 2*i);
      double img = in_feats(frm_idx, 2*i+1);
      energy[i] = sqrt(real * real + img * img);
    }

    for (size_t bin = 0; bin < mel_weights.size(); bin++) {
      double mel_energy = 0.0;
      for (int f = 0; f < mel_weights[bin].size(); f++) {
        mel_energy += mel_weights[bin][f] * energy[f];
      }
      if (do_log) {
        mel_energy = log(mel_energy);
      }
      out_feats(frm_idx, bin) = mel_energy;
    }
  }
  //  END_LAB
}

/** Module for doing discrete cosine transform. **/
// change to google name style
void FrontEnd::do_dct(const matrix<double>& in_feats,
                      matrix<double>& out_feats) const {
  //  Number of DCT coefficients to output.
  int num_coeffs = get_int_param(params_, "dct.coeffs", 12);
  int in_frame_cnt = in_feats.size1();
  int in_dim_cnt = in_feats.size2();
  int out_dim_cnt = num_coeffs;

  //  Allocate output matrix and fill with zeros.
  out_feats.resize(in_frame_cnt, out_dim_cnt);
  out_feats.clear();

  //  BEGIN_LAB
  //
  //  Input:
  //      The matrix "in_feats", holding the output of mel-binning.
  //
  //      in_feats(0 .. (in_frame_cnt - 1), 0 .. (in_dim_cnt - 1))
  //
  //  Output:
  //      The matrix "out_feats", which should contain the result of
  //      applying the DCT.
  //
  //      out_feats(0 .. (in_frame_cnt - 1), 0 .. (out_dim_cnt - 1))
  //
  //      "out_feats" has been allocated to be of the correct size.
  //
  //  See "in_frame_cnt", "in_dim_cnt", and "out_dim_cnt" above
  //  for quantities you will need for this computation.
  int num_mel_bins = in_dim_cnt;
  std::vector<std::vector<double> > dct_weights(num_coeffs);
  for (size_t dct_bin = 0; dct_bin < dct_weights.size(); dct_bin++) {
    dct_weights[dct_bin].resize(num_mel_bins);
    for (int mel_bin = 0; mel_bin < num_mel_bins; mel_bin++) {
      dct_weights[dct_bin][mel_bin] = cos(M_PI * (dct_bin + 1) *
                                      (mel_bin + 1.0 / 2) / num_mel_bins);
    }
  }

  double scale = sqrt(2.0 / num_mel_bins);
  for (int frm_idx= 0; frm_idx < in_frame_cnt; frm_idx++) {
    for (int dct_bin = 0; dct_bin < num_coeffs; dct_bin++) {
      double dct_sum = 0.0;
      for (int mel_bin = 0; mel_bin < num_mel_bins; mel_bin++) {
        dct_sum += dct_weights[dct_bin][mel_bin] * in_feats(frm_idx, mel_bin);
      }
      out_feats(frm_idx, dct_bin) = dct_sum * scale;
    }
  }
  //  END_LAB
}

/** Main signal processing routine.
 *   Calls each signal processing module in turn, unless
 *   parameter says not to.
 **/
void FrontEnd::get_feats(const matrix<double>& inAudio,
                         matrix<double>& out_feats) const {
  if (get_bool_param(params_, "frontend.null", false)) {
    out_feats = inAudio;
    return;
  }
  matrix<double> curFeats(inAudio);
  if (get_bool_param(params_, "frontend.window", true)) {
    do_window(curFeats, out_feats);
    out_feats.swap(curFeats);
  }
  if (get_bool_param(params_, "frontend.fft", true)) {
    do_fft(curFeats, out_feats);
    out_feats.swap(curFeats);
  }
  if (get_bool_param(params_, "frontend.melbin", true)) {
    do_melbin(curFeats, out_feats);
    out_feats.swap(curFeats);
  }
  if (get_bool_param(params_, "frontend.dct", true)) {
    do_dct(curFeats, out_feats);
    out_feats.swap(curFeats);
  }
  out_feats.swap(curFeats);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
