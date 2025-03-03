#include <NDP.h>
#include <NDP_utils.h>
#include <Arduino.h>
#include "TinyML_init.h"
#include "NDP_init.h"
#include "NDP_loadModel.h"
#include "SAMD21_init.h"
#include "SAMD21_lowpower.h"

typedef enum {
  MATCH_NONE = 0,
  BLUE
} classifier_match_t;

#define NDP_MICROPHONE 0
#define NDP_SENSOR 1

static volatile classifier_match_t s_match;
static void ndp_isr(void) {
   int match_result = NDP.poll();
    Serial.print("Classifier detected: ");
    Serial.println(match_result);  // See what number is being returned

    s_match = (classifier_match_t)match_result;
}

void service_ndp() {
  switch (s_match) {
    case MATCH_NONE:
      break;

    case BLUE:
      Serial.println("Wake word detected!");
      digitalWrite(LED_BLUE, HIGH);
      digitalWrite(LED_GREEN, HIGH);
      delay(1000);
      digitalWrite(LED_BLUE, LOW);
      digitalWrite(LED_GREEN, LOW);
      s_match = MATCH_NONE;  // Reset after handling
      break;

    default:
      Serial.println("Unknown classifier result.");
      break;
  }
}

void setup(void) {
  // Initialize the SAMD21 host processor
  SAMD21_init(0);
  // load the the board with the model "x" it was shipped from the \
  // factory
  NDP_init("ei_model.bin", NDP_MICROPHONE);
  //NDP_init("google10.bin", NDP_MICROPHONE);
  // The NDP101 will wake the SAMD21 upon detection
  attachInterrupt(NDP_INT, ndp_isr, HIGH);
  // Prevent the internal flash memory from powering down in sleep mode
  NVMCTRL->CTRLB.bit.SLEEPPRM = NVMCTRL_CTRLB_SLEEPPRM_DISABLED_Val;
  // Select STANDBY for sleep mode
  SCB->SCR |= SCB_SCR_SLEEPDEEP_Msk;
}

void loop() {
  // Put various peripheral modules into low power mode before entering standby.
  adc_disable();        // See SAMD21_lowpower.h
  usb_serial_disable(); // See note above about USB communications
  systick_disable();
  // Complete any memory operations and enter standby mode until an interrupt
  // is recieved from the NDP101
  __DSB();
  __WFI();
  // Arrive here after waking and having called ndp_isr().  Re-enable various
  // peripheral modules before servicing the NDP classifier data.
  systick_enable();
  usb_serial_enable();
  Serial.println("Debug: Awake from deep sleep");
  adc_enable();
  // process the classifier data from the NDP
  service_ndp();
  // loop (and return immediately to standby mode)
}
