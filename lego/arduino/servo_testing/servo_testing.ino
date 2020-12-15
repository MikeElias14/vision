// Sero setup
#include <Servo.h>
Servo shoulder;
Servo elbow;
Servo wrist;

int defaultShoulder = 90; // Shoulder is 90-180
int defaultElbow = 180; // Vertical is 180
int defaultWrist = 0;

int shoulderAngle = 110;
int elbowAngle = 90;
int wristAngle = 0;

// Gyro setup
#include <Wire.h>
#include <MPU6050.h>

MPU6050 mpu;

String command;

void setup() {  
  // Servo init
  Serial.begin(115200);
  shoulder.attach(9);
  elbow.attach(10);
  wrist.attach(11);
  
  shoulder.write(defaultShoulder);
  elbow.write(defaultElbow);
  wrist.write(defaultWrist);
  delay(3000);  
}

void loop() {
  
}

  
