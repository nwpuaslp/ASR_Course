# if you install openfst and dot, you could visualize the fsm in pdf
# some fsm file contains more than one fsm, you need to split it by hand before using this cmd.
grep -v '#' p018k1.noloop.fsm|fstcompile --osymbols=p018k2.syms|fstdraw -portrait --osymbols=p018k2.syms|dot -Tpdf -o p018k1.noloop.pdf