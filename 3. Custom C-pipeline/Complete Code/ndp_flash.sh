
#!/bin/bash
echo "Flashing to NDP101..."

gcc src/*.c -o ndp_model -I./include -I/opt/homebrew/include -L/opt/homebrew/lib -lportaudio -lfftw3 -lm
./ndp_model
