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

  shoulder.write(shoulderAngle);
  elbow.write(elbowAngle);
  wrist.write(wristAngle);
  delay(2000);

  // Gyro init
  Serial.println("Initialize MPU6050");

  while(!mpu.begin(MPU6050_SCALE_2000DPS, MPU6050_RANGE_2G))
  {
    Serial.println("Could not find a valid MPU6050 sensor, check wiring!");
    delay(500);
  }

  // Set accelerometer offsets
//   mpu.setAccelOffsetX();
//   mpu.setAccelOffsetY();
//   mpu.setAccelOffsetZ();
  
  checkSettings();  
}

void loop() {
  
  // Read command from serial
  command = Serial.readString();
  Serial.println(command);

  /* 
   * Forward (w) or back (s) control:
   * Do this by moving the shoulder, and adjusting the elbow
   * to keep the Z constant.
   */

  /* 
   * Up (e) or Down (q) control:
   * Do this by moving the Elbow, and adjusting the Shoulder
   * to keep the plane constant.
   */

   if ( command.indexOf("w") > -1 ) {
     moveOut();
   } 
   else if ( command.indexOf("s") > -1 ) {
     moveIn();
   } 
   else if ( command.indexOf("e") > -1 ) {
     moveUp();
   } 
   else if ( command.indexOf("q") > -1 ) {
     moveDown();
   }

   // Reset to vertical and stop
   if ( command.indexOf("stop") > -1 ) {
    shutOff();
   }
  
  // Adjust wrist to stay vertical
  adjustWrist();

  // Print values
//  Serial.print("Shoulder = ");
//  Serial.print(shoulderAngle);
//  Serial.print("; Elbow = ");
//  Serial.print(elbowAngle);
//  Serial.print("; Wrist = ");
//  Serial.print(wristAngle);
//  Serial.print("; Roll = ");
//  Serial.print(getRoll());
//  Serial.println();

  delay(50);
}

void moveOut() {
  Serial.println("Move Out");
  // Get origional axis
  Vector accel = getAccel();

  // Move elbow
  if ( elbowAngle < 170 ) {
    elbowAngle += 5;
    elbow.write(elbowAngle);

    // Adjust shoulder
    adjustVertical(accel.ZAxis);
  } 
}

void moveIn() {
  Serial.println("Move In");
  // Get origional axis
  Vector accel = getAccel();

  // Move elbow
  if ( elbowAngle > 10 ) {
    elbowAngle -= 5;
    elbow.write(elbowAngle);

    // Adjust Elbow
    adjustVertical(accel.ZAxis);
  } 
}

void moveUp() {
  Serial.println("Move Up");
  // Get origional axis
  Vector accel = getAccel();

  // Move shoulder
  if ( shoulderAngle > 90 ) {
    shoulderAngle -= 5;
    shoulder.write(shoulderAngle);

    // Adjust elbow
    adjustHorizontal(accel.XAxis, accel.YAxis);
  } 
}

void moveDown() {
  Serial.println("Move Down");
  // Get origional axis
  Vector accel = getAccel();

  // Move shoulder
  if ( shoulderAngle < 170 ) {
    shoulderAngle += 5;
    shoulder.write(shoulderAngle);

    // Adjust elbow
    adjustHorizontal(accel.XAxis, accel.YAxis);
  } 
}

void shutOff() {
  shoulder.write(defaultShoulder);
  elbow.write(defaultElbow);
  wrist.write(defaultWrist);
  delay(3000);

  
  shoulderAngle = 110;
  elbowAngle = 90;
  wristAngle = 0;

  shoulder.write(shoulderAngle);
  elbow.write(elbowAngle);
  wrist.write(wristAngle);
  delay(2000);
}

int getRoll() {
  int thisRoll;
  Vector normAccel = mpu.readNormalizeAccel();
  thisRoll = (atan2(normAccel.YAxis, normAccel.ZAxis)*180.0)/M_PI;
  return thisRoll;
}

Vector getAccel() {
  Vector rawAccel = mpu.readRawAccel();
  return rawAccel;
}

// Adjust elbow to maintain origional x and y positioning - always straighten
void adjustHorizontal(double origX, double origY) {
  Vector accel = getAccel();

  int origDist = int(sqrt(origX*origX + origY*origY));

  if ( sqrt(accel.XAxis*accel.XAxis + accel.YAxis*accel.YAxis) < origDist && elbowAngle < 175 ) {
    while ( sqrt(accel.XAxis*accel.XAxis + accel.YAxis*accel.YAxis) < origDist && elbowAngle < 175 ) {
      elbowAngle += 1;
      elbow.write(elbowAngle);
      delay(10);
      accel = getAccel();
    }
  }
  else if ( sqrt(accel.XAxis*accel.XAxis + accel.YAxis*accel.YAxis) > origDist && elbowAngle < 175 ) {
    while ( sqrt(accel.XAxis*accel.XAxis + accel.YAxis*accel.YAxis) > origDist && elbowAngle < 175 ) {
      elbowAngle += 1;
      elbow.write(elbowAngle);
      delay(10);
      accel = getAccel();
    }
  }

  Serial.println("Adjusted Horizontal: ");
  Serial.print("Origional Dist = ");
  Serial.print(origDist);
  Serial.print("; New Dist = ");
  Serial.println(int(sqrt(accel.XAxis*accel.XAxis + accel.YAxis*accel.YAxis)));
}


// Adjust shoulder to maintain origional Z positioning
void adjustVertical(double origZ) {
  Vector accel = getAccel();

  if ( accel.ZAxis > origZ && shoulderAngle < 175 ) {
    while ( accel.ZAxis > origZ && shoulderAngle < 175 ) {
      shoulderAngle += 1;
      shoulder.write(shoulderAngle);
      delay(10);
      accel = getAccel();
    }
  }
  else if ( accel.ZAxis < origZ && shoulderAngle > 90 ) {
    while ( accel.ZAxis < origZ && shoulderAngle > 90 ) {
      shoulderAngle -= 1;
      shoulder.write(shoulderAngle);
      delay(10);
      accel = getAccel();
    }
  }

  Serial.println("Adjusted Vertical: ");
  Serial.print("Origional Z = ");
  Serial.print(origZ);
  Serial.print("; New Z = ");
  Serial.println(accel.ZAxis);
}


void adjustWrist() {
  int roll = getRoll();
  
  if (roll <= -100 && wristAngle > 10 ) {
    while ( roll <= -100 && wristAngle > 10 ) {
      wristAngle -= 1;
      wrist.write(wristAngle);
      delay(10);
      roll = getRoll();
    }
  }
  else if ( roll >= -80 && wristAngle < 170 ) {
    while ( roll >= -80 && wristAngle < 170 ) {
      wristAngle += 1;
      wrist.write(wristAngle);
      delay(10);
      roll = getRoll();
    }
  }
}


void checkSettings()
{
  Serial.println();
  
  Serial.print(" * Sleep Mode:            ");
  Serial.println(mpu.getSleepEnabled() ? "Enabled" : "Disabled");
  
  Serial.print(" * Clock Source:          ");
  switch(mpu.getClockSource())
  {
    case MPU6050_CLOCK_KEEP_RESET:     Serial.println("Stops the clock and keeps the timing generator in reset"); break;
    case MPU6050_CLOCK_EXTERNAL_19MHZ: Serial.println("PLL with external 19.2MHz reference"); break;
    case MPU6050_CLOCK_EXTERNAL_32KHZ: Serial.println("PLL with external 32.768kHz reference"); break;
    case MPU6050_CLOCK_PLL_ZGYRO:      Serial.println("PLL with Z axis gyroscope reference"); break;
    case MPU6050_CLOCK_PLL_YGYRO:      Serial.println("PLL with Y axis gyroscope reference"); break;
    case MPU6050_CLOCK_PLL_XGYRO:      Serial.println("PLL with X axis gyroscope reference"); break;
    case MPU6050_CLOCK_INTERNAL_8MHZ:  Serial.println("Internal 8MHz oscillator"); break;
  }
  
  Serial.print(" * Accelerometer:         ");
  switch(mpu.getRange())
  {
    case MPU6050_RANGE_16G:            Serial.println("+/- 16 g"); break;
    case MPU6050_RANGE_8G:             Serial.println("+/- 8 g"); break;
    case MPU6050_RANGE_4G:             Serial.println("+/- 4 g"); break;
    case MPU6050_RANGE_2G:             Serial.println("+/- 2 g"); break;
  }  

  Serial.print(" * Accelerometer offsets: ");
  Serial.print(mpu.getAccelOffsetX());
  Serial.print(" / ");
  Serial.print(mpu.getAccelOffsetY());
  Serial.print(" / ");
  Serial.println(mpu.getAccelOffsetZ());
  
  Serial.println();
}


  
