ROOTFLAGS = $(shell root-config --cflags)
ROOTLIBS = $(shell root-config --libs)

CXXFLAGS = -Wall -O2 $(ROOTFLAGS)
CXXLIBS = $(ROOTLIBS)

all: MAIKo merge_rootfile

MAIKo: MAIKo.o analysis.o database.o dataset.o kinema.o nuclear.o bethelib.o kinelib.o bethe_doi.o
	g++ -o $@ database.o dataset.o kinema.o nuclear.o bethelib.o kinelib.o analysis.o bethe_doi.o MAIKo.o $(CXXLIBS) -O2

merge_rootfile: merge_rootfile.o
	g++ -o $@ merge_rootfile.o $(CXXLIBS) -O2

.cc.o:
	g++ -c $(CXXFLAGS) $<

.c.o:
	g++ -c $(CXXFLAGS) $<

.cpp.o:
	g++ -c $(CXXFLAGS) $<

clean:
	$(RM) MAIKo *.o
