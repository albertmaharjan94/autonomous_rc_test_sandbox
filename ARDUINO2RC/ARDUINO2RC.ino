#include <Servo.h>
Servo MOTOR;     // create servo object to control the MOTOR
Servo DIRECTION;


int motor_pin = 7;
int direction_pin = 6;

int dead_zone = 1490;

int minAngle = 1147; //60 degree
//int minAngle = 1257; //60 degree
int midAngle = 1467; //90 degree
int maxAngle = 1787; //120 degree
//int maxAngle = 1677; //120 degree

int min_speed = 1430;
int max_speed = 1570;
int OFFSET = 50;
int _delay = 10;

void writeSpeed();

int current_speed = dead_zone;
int angleGiven = midAngle;

boolean status = true;

void setup() {
  MOTOR.attach(motor_pin,1000,2000); // MOTOR d-pins on 11, with input value from 1000-2000
  DIRECTION.attach(direction_pin);
  
  Serial.begin(19200);
  DIRECTION.writeMicroseconds(midAngle); 
  MOTOR.writeMicroseconds(current_speed);
  
  Serial.println("Initializing for 5 seconds");
  delay(2000);  // Send the signal to the ESC
  
}
void loop() {
  
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    char *string = data.c_str();
    char delim[] = "#";
    char * token = strtok(string, delim);
    angleGiven = map(atoi(token),0,60,minAngle,maxAngle);
    while (token != NULL) {
      token = strtok(NULL, delim);
      if (token != NULL) {
        if(atoi(token) == 10){
           current_speed = 1539;
        }else if(atoi(token) == -10){
          current_speed = 1448;
        }else{
          current_speed = dead_zone;
        }
      }
    }
  }
  writeSpeed();
  DIRECTION.writeMicroseconds(angleGiven); 
  Serial.print(angleGiven); Serial.print(" "); Serial.println(current_speed);
  delay(12.5);
  
}
void writeSpeed(){
  if(current_speed == dead_zone){
    current_speed = dead_zone;
    MOTOR.writeMicroseconds(current_speed);
  }
  else if(current_speed > dead_zone){
    if (status == true){
      if(current_speed >= max_speed){
        current_speed = max_speed;
      }
      MOTOR.writeMicroseconds(current_speed);
      delay(_delay);
    }else{
      int tmp = current_speed;
      current_speed = dead_zone;
//      enable esc doesnt need extra paramters to forward after reverse
      MOTOR.writeMicroseconds(current_speed);
      delay(25);
      while(current_speed <= max_speed ){
        if(current_speed >= max_speed){
          current_speed = max_speed;
          break;
        }
        if(current_speed >=tmp){
          current_speed = tmp;
          break;
        }
        MOTOR.writeMicroseconds(current_speed);
        current_speed += 20;
        delay(_delay);
      } 
    }
    status = true;
  }
  else if(current_speed < dead_zone){
    if (status == false){
      if(current_speed <= min_speed){
        current_speed = min_speed;
      }
      MOTOR.writeMicroseconds(current_speed);
      delay(_delay);
    }else{
      int tmp = current_speed;
      current_speed = dead_zone;
//      enable esc , needs extra paramters to reverse after forward
      MOTOR.writeMicroseconds(current_speed);
      delay(25);
      MOTOR.writeMicroseconds(current_speed-OFFSET);
      delay(80);
      MOTOR.writeMicroseconds(current_speed);
      delay(80);
      while(current_speed >= min_speed){
        if(current_speed <= min_speed){
          current_speed = min_speed;
          break;
        } 
        if(current_speed <= tmp){
          current_speed = tmp;
          break;
        }
        MOTOR.writeMicroseconds(current_speed);
        delay(_delay);
        current_speed -= 20;
      } 
    }
    status = false;
  }
}
