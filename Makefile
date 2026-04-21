# Makefile for le.c — planetary mini-weights engine
#
# Targets:
#   make           build ./le
#   make test      build & run tests/test_le
#   make run       build and run with --meta 2 --seed 42
#   make clean     remove built artifacts

CC      ?= cc
CFLAGS  ?= -O2 -Wall
LDLIBS  ?= -lm

.PHONY: all test run clean

all: le

le: le.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

tests/test_le: tests/test_le.c le.c
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-unused-variable -o $@ tests/test_le.c $(LDLIBS)

test: tests/test_le
	./tests/test_le

run: le
	./le --meta 2 --seed 42

clean:
	rm -f le tests/test_le
