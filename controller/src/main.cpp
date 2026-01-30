#include <Arduino.h>

int state[3] = {0,0,0};
const int MAX_LEFT = 10;
const int MAX_RIGHT = 10;
const int MAX_MAIN = 10;

int mapCommand(byte cmd){
  switch(cmd){
    case 0x00: return -1;
    case 0x01: return 0;
    case 0x02: return 1;
    default: return 0;
  }
}

void setup(){
  Serial.begin(115200);
  while(!Serial);
}

void loop(){
  if(Serial.available() > 0){
    byte b0 = Serial.read();

    // Reset
    if(b0 == 0xFF){
      if(Serial.available() >= 3){
        state[0] = Serial.read();
        state[1] = Serial.read();
        state[2] = Serial.read();
      }
    }
    // Normal action
    else{
      if(Serial.available() >= 2){
        byte b1 = Serial.read();
        byte b2 = Serial.read();

        state[0] = constrain(state[0] + mapCommand(b0), 0, MAX_LEFT);
        state[1] = constrain(state[1] + mapCommand(b1), 0, MAX_RIGHT);
        state[2] = constrain(state[2] + mapCommand(b2), 0, MAX_MAIN);
      }
    }

    // Always write back 3 bytes
    Serial.write(state[0]);
    Serial.write(state[1]);
    Serial.write(state[2]);
    Serial.flush(); // ensure bytes are sent
  }
}
