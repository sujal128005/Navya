
  // --- Configuration ---
const int rcThrottlePin = 3;  // Receiver Channel 2 (Throttle Input)

const int throttleOutPin = 9; // PWM pin to Motor Driver Speed/Enable
const int reverseOutPin = 8;  // Single wire Direction pin (Connected to Pin 8 now)

// Variables
unsigned long throttleValue = 0;
unsigned long lastLogTime = 0; 

void setup() {
  pinMode(rcThrottlePin, INPUT);
  
  pinMode(throttleOutPin, OUTPUT);
  pinMode(reverseOutPin, OUTPUT);
  
  Serial.begin(9600);
  
  // Safety Init: Stop motor completely on startup
  analogWrite(throttleOutPin, 0); 
  digitalWrite(reverseOutPin, LOW);
  
  Serial.println("Rear-Wheel Only Mode Active.");
}

void loop() {
  // 1. Read the Throttle Signal from the RC Receiver
  throttleValue = pulseIn(rcThrottlePin, HIGH, 30000);

  // --- DEBUG TOOLS ---
  if (millis() - lastLogTime > 200) {
    Serial.print("Throttle Pulse: "); Serial.println(throttleValue);
    lastLogTime = millis();
  }

  // 2. Handle Throttle & Direction Logic
  if (throttleValue > 800 && throttleValue < 2200) {
    
    // FORWARD
    if (throttleValue > 1550) {
      digitalWrite(reverseOutPin, LOW); // Your working configuration high state
      
      int speed = map(throttleValue, 1550, 1900, 50, 255); 
      speed = constrain(speed, 0, 255);
      analogWrite(throttleOutPin, speed);
    } 
    // REVERSE
    else if (throttleValue < 1450) {
      digitalWrite(reverseOutPin, HIGH);  // Your working configuration low state
      
      int speed = map(throttleValue, 1450, 1100, 50, 255); 
      speed = constrain(speed, 0, 255);
      analogWrite(throttleOutPin, speed);
    } 
    // DEADZONE - INSTANT STOP
    else {
      analogWrite(throttleOutPin, 0); 
    }
  } 
  else {
    // Safety Stop if remote goes out of range or disconnects
    analogWrite(throttleOutPin, 0);
  }
}
