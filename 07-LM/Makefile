
########################################################################
#   Preamble.
########################################################################

ifeq ($(OPTFLAGS),)
	OPTFLAGS = -g
endif

CXX = g++
CPPFLAGS = -I/user1/faculty/stanchen/pub/boost
CXXFLAGS = -Wall
LDFLAGS = $(OPTFLAGS)
LDLIBS = -lm

CXXFLAGS += $(OPTFLAGS)

#   GNU make's default rule uses $(CC) for linking
LINK.o = $(CXX) $(LDFLAGS) $(TARGET_ARCH)


########################################################################
#   Rules.
########################################################################

all : lab3_lm

clean:
	rm -f lab3_lm *.o Lab3Lm.class LangModel.class

util.o : util.C util.H

lang_model.o : lang_model.C lang_model.H util.H

main.o : main.C

lab3_lm.o : lab3_lm.C util.H lang_model.H

lab3_lm : lab3_lm.o util.o lang_model.o main.o


########################################################################
#   
########################################################################


