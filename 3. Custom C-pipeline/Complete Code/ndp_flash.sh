
#!/bin/bash
echo "Flashing to NDP101..."

gcc src/*.c -o ndp_model -I./include -I/opt/homebrew/include -L/opt/homebrew/lib -lportaudio -lfftw3 -lm
#./ndp_model 1 ./wav_files/one.wav
./ndp_model 2
python3 ./scripts/spectrogram_draw.py
