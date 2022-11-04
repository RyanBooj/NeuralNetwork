# Default
CXX = g++
CFLAGS = -g -Wall

# Files
SRC_FILES = Network.cpp Neuron.cpp TransferFns.cpp
OBJ_FILES = Network.o Neuron.o TransferFns.o

# Valgrind
VLGRND_FLAGS = --leak-check=yes --show-leak-kinds=all --track-origins=yes 

# Default target
all: $(OBJ_FILES) $(SRC_FILES)

# Implicit version of:
# 	Implicit compilation rules
# 	Network.o: Network.cpp
# 	Layer.o: Layer.cpp
# 	Neuron.o: Neuron.cpp
$(OBJ_FILES): %.o: %.cpp

test: $(TEST_OBJ) $(OBJ_FILES) test.cpp
	$(CXX) $(CFLAGS) -o test test.cpp $(SRC_FILES)

valgrind: test.o $(OBJ_FILES)
	valgrind $(VLGRND_FLAGS) ./test

clean:
	rm ./*.o