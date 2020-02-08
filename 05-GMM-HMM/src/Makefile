
########################################################################
#   Preamble.
########################################################################

ifeq ($(OPTFLAGS),)
	OPTFLAGS = -g
endif

CXX = g++
CXXFLAGS = -Wall -std=c++11
LDFLAGS = $(OPTFLAGS)
LDLIBS = -lm

CXXFLAGS += $(OPTFLAGS)

#   GNU make's default rule uses $(CC) for linking
LINK.o = $(CXX) $(LDFLAGS) $(TARGET_ARCH)

# change this to the path of your last assignment

########################################################################
#   Rules.
########################################################################

all : lab2_vit lab2_train lab2_fb

clean:
	rm -f lab2_vit lab2_train lab2_fb *.o front_end.o

util.o : util.C util.H

gmm_util.o : gmm_util.C gmm_util.H util.H

main.o : main.C

lab2_vit.o : lab2_vit.C lab2_vit.H util.H front_end.H

lab2_train.o : lab2_train.C util.H front_end.H gmm_util.H

lab2_fb.o : lab2_fb.C util.H front_end.H gmm_util.H

lab2_vit : lab2_vit.o util.o front_end.o main.o

lab2_train : lab2_train.o util.o front_end.o gmm_util.o main.o

lab2_fb : lab2_fb.o util.o front_end.o gmm_util.o main.o

front_end.o: front_end.C util.H front_end.H
########################################################################
#   
########################################################################


