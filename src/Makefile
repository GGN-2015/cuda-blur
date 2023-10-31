NVCC=nvcc
OBJ=cuda-blur
SRC=main.cu

all: $(OBJ)

$(OBJ): $(SRC)
	$(NVCC) -o $(OBJ) $(SRC)

run: all
	./$(OBJ)

clean:
	rm -f $(OBJ)

.PHONY: all run clean