# Intership Progression

## 1. Understanding NDP101

- [X] Understanding the configuration and how to interact with NDP101 (Image 1.10 interaction with the host processor consisting in Arduino MKRZero)
- [X] Training a word label to see how to create a custom model (used Edge Impulse)
- [X] Creating the code to handle hardware configuration of Arduino MKRZero (Code developed by ARDUINO IDE and already built libraries)

Difficulties:
* Flash Memory address counting problem, after some testing and uploads the device didn’t allow any upload because it looked like reading in binary file 0xFF. To resolve this I had to totally reset with a command send to host processor (command: F) this like set the status at the original status
* At some point the device was stuck in a blue light, different from the voice recognition one, this raised due to a model that wasn’t found, so when having a custom model is necessary to provide in Arduino code the path to the desired model, corresponding to the composed firmware

What Learned:
* Understand the structure of the device and the interfacing
* Understanding how to manage custom created models and upload them on the device
* Understand the default libraries to work with Arduino IDE with the objective to interface with the hardware

## 2. Specifics, Interactions and Control
