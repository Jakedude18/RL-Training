#include <Arduino.h>


// Linear Actuator Motor Driver Test
// Compatible with Arduino-style boards
// Tests forward, reverse, speed ramp, and stop

const int PWM_PIN = 1;      // Connect to EN / PWM on motor driver
const int DIR_PIN_1 = 2;    // IN1
// const int DIR_PIN_2 = 8;    // IN2

const int MAX_SPEED = 255;  // PWM max (0-255)


// ----------------------
// Motor Control Functions
// ----------------------

void extendActuator() {
  digitalWrite(DIR_PIN_1, HIGH);
  // digitalWrite(DIR_PIN_2, LOW);
}

void retractActuator() {
  digitalWrite(DIR_PIN_1, LOW);
  // digitalWrite(DIR_PIN_2, HIGH);
}

void stopMotor() {
  digitalWrite(DIR_PIN_1, LOW);
  // digitalWrite(DIR_PIN_2, LOW);
  analogWrite(PWM_PIN, 0);
}

// Smooth ramp prevents sudden current spikes
void rampSpeedUp() {
  for (int speed = 0; speed <= MAX_SPEED; speed += 5) {
    analogWrite(PWM_PIN, speed);
    delay(40);
  }
}



void setup() {
  Serial.begin(9600);

  pinMode(PWM_PIN, OUTPUT);
  pinMode(DIR_PIN_1, OUTPUT);
  // pinMode(DIR_PIN_2, OUTPUT);

  stopMotor();

  Serial.println("Linear Actuator Motor Test Starting...");
}

void loop() {
  Serial.println("Extending actuator...");
  extendActuator();
  rampSpeedUp();
  delay(4000);

  Serial.println("Stopping...");
  stopMotor();
  delay(2000);

  Serial.println("Retracting actuator...");
  retractActuator();
  rampSpeedUp();
  delay(4000);

  Serial.println("Stopping...");
  stopMotor();
  delay(4000);
}

