# Intership Progression

## 1. Understanding NDP101

To Do:<br>

- [X] Understanding the configuration and how to interact with NDP101 (Image 1.10 interaction with the host processor consisting in Arduino MKRZero)
- [X] Training a word label to see how to create a custom model (used Edge Impulse)
- [X] Creating the code to handle hardware configuration of Arduino MKRZero (Code developed by ARDUINO IDE and already built libraries)

Difficulties:<br>

* Flash Memory address counting problem, after some testing and uploads the device didn’t allow any upload because it looked like reading in binary file 0xFF. To resolve this I had to totally reset with a command send to host processor (command: F) this like set the status at the original status
* At some point the device was stuck in a blue light, different from the voice recognition one, this raised due to a model that wasn’t found, so when having a custom model is necessary to provide in Arduino code the path to the desired model, corresponding to the composed firmware

What Learned:<br>

* Understand the structure of the device and the interfacing
* Understanding how to manage custom created models and upload them on the device
* Understand the default libraries to work with Arduino IDE with the objective to interface with the hardware

## 2. Specifics, Interactions and Control

To Do:<br>

- [X] What is the power consumption of NDP101? (Audio Recognition power consumption 100 uW)
- [X] What kind of features can be extracted using NDP101? (audio-based features (spectrogram, event detection, raw audio)), not accessible SDK
    - [X] What is the internal architecture? How can we program it? (Audio capture and processing, neural network, small SRAM of 112 KB, interfacing with the outside)
    - [X] How much memory does it have? What kind of operations? (Operations of NDP101 are model processing and uploading)
- [X] How can we program it w/o Arduino? Is there any SDK? That we can use pure C. (Problem with NDA, possible in theory but not modifiable) 
    - [X] How can I program host processor? Is it ARM-Cortex? (Arduino IDE it has a firmware with NDP101 and is Cortex-M0+)
    - [X] How can I control NDP101? Driver? (We can't program NDP101 directly, but we can handle the communications with it)
    - [X] Can I connect another MCU to NDP101?  (Yes, we can do that configuring the GPIOs with and SPI, I2C or UART communication, depending on the necessity)
    - [X] What is PDM connection? (Signal represented in binary, like a stream of bits)
- [X] Is there any code in Edge Impulse Filmware repository for NDP101?
    - [X] Is that code usable or useful? (Yes, there is some code underlining the possibility of communication modification or another about memory management, more details on file 2. Notes)
- [X] Go indepth with Speaker Recognition for Deeply Embedded IoT Devices (https://mlsysbook.ai/contents/labs/seeed/xiao_esp32s3/kws/kws.html)

Difficulties:<br>

* Problem with SDK of Syntiant NDP101 and because of this the processor is not programmable, but the result may be handled in destination
* Impossibility of features specifics extraction, because the SDK can't allow Syntiant NDP101 introspection of the neural network implementation, the results may be obtained via Edge Impulse, but is not explicited the implementation

What Learned:<br>

* Possibility of interaction with other MCUs possible
* Better firmware Edge Impulse understanding, including the interaction implementation with the host processor
* Better understanding of the structure of the board
