
//  $Id: main.C,v 1.1 2009/10/01 13:52:24 stanchen Exp $

#include <iostream>
#include <stdexcept>

using namespace std;

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

extern void main_loop(const char** argv);

int main(int argc, const char** argv) {
  try {
    main_loop(argv);
  } catch (exception& xc) {
    cerr << "Error: " << xc.what() << endl;
    return -1;
  }
  return 0;
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
