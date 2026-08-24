CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra
BUILD_DIR := build

.PHONY: all clean test

all: $(BUILD_DIR)/attribution_engine $(BUILD_DIR)/test_black_scholes

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/black_scholes.o: src/black_scholes.cpp src/black_scholes.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c src/black_scholes.cpp -o $(BUILD_DIR)/black_scholes.o

$(BUILD_DIR)/attribution_engine: src/attribution_main.cpp $(BUILD_DIR)/black_scholes.o | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) src/attribution_main.cpp $(BUILD_DIR)/black_scholes.o -o $(BUILD_DIR)/attribution_engine

$(BUILD_DIR)/test_black_scholes: tests/test_black_scholes.cpp $(BUILD_DIR)/black_scholes.o | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) tests/test_black_scholes.cpp $(BUILD_DIR)/black_scholes.o -o $(BUILD_DIR)/test_black_scholes

test: $(BUILD_DIR)/test_black_scholes
	$(BUILD_DIR)/test_black_scholes

clean:
	rm -rf $(BUILD_DIR)
