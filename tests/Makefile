# Uncomment one of the following to switch between debug and opt mode
OPT = -O3 -DNDEBUG -std=c++17
#OPT = -g -ggdb -fsanitize=address -fno-omit-frame-pointer -Wextra -fsanitize=undefined

CXXFLAGS += -fno-strict-aliasing -Wall -std=c++11 -I. -I../src/ \
-I../src/xorfilter -I../src/ribbon \
$(OPT)

UNAME_P := $(shell uname -p)
ifeq ($(UNAME_P),x86_64)
        CXXFLAGS += -march=native
else
        CXXFLAGS +=
endif
LDFLAGS = -Wall -Wextra

HEADERS = $(wildcard ../src/*.h \
    ../src/xorfilter/*.h  ../src/ribbon/*.h \
    ) *.h

.PHONY: all


RFILTER = test_rfilter

all: $(RFILTER)
test_rfilter: fast_ribbonfilter_test.cc Makefile
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	/bin/rm -f $(RFILTER)
