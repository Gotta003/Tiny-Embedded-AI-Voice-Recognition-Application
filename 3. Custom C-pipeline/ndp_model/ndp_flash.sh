
#!/bin/bash
echo "Flashing to NDP101..."

gcc main.c mfe.c dnn.c -o ndp_model -I/opt/homebrew/include -L/opt/homebrew/lib -lportaudio -lfftw3 -lm
./ndp_model
