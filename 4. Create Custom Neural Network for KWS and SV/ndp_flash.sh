
#!/bin/bash
echo "Flashing to NDP101..."

python3 ./scripts/jsonReader.py
gcc src/*.c -o ndp_model -I./include -I/opt/homebrew/include -L/opt/homebrew/lib -lportaudio -lfftw3 -lm
#./ndp_model 0
./ndp_model 1 ./wav_files/matteo.wav
#./ndp_model 2
python3 ./scripts/spectrogram_draw.py
