
#!/bin/bash
echo "Flashing to NDP101..."

gcc src/main.c src/mfe.c src/dnn.c -o ndp_model -I./include -I/opt/homebrew/include -L/opt/homebrew/lib -lportaudio -lfftw3 -lm
./ndp_model
