/* 
This code controls an Arduino which outputs a digital signal on two PINs.

*/

//Arduino ports
#define triggerPin A2
#define triggerPin2 9
#define MOTPin 2
#define PAPin 5
int digitalTrigger = 0;

typedef enum{
  OnOn,
  OnOff,
  OffOn,
  OffOff
} RunType;

#define OperationMode OffOff //operation of if there is overlap between the MOT and PA beam before/after the PA beam turns (On means an overlap, Off means a gap)

float cycle_frequency = 1000;  //duty cycle frequency in Hz
float MOT_uptime = 60;   //% of duty cycle where MOT is on (and thus PA beam is off)
float PA_uptime = 40;    //% of duty cycle where MOT is on (and thus PA beam is off)
float PA_prefrac = 0.;   //% of duty cycle offset between end of MOT and start of PA
float PA_postfrac = 0.;  //% of duty cycle offset between end of PA and start of MOT

const float MOTtime = (1000*MOT_uptime)/(100*cycle_frequency)*1000;  //duration of MOT uptime in s
const float PAtime = (1000*PA_uptime)/(100*cycle_frequency)*1000;    //duration of MOT uptime in s

const float PA_pretime = 1000000*PA_prefrac/100/cycle_frequency;
const float PA_posttime = 1000000*PA_postfrac/100/cycle_frequency;

void setup() {
  pinMode(triggerPin, INPUT);
  pinMode(triggerPin2, INPUT); 
  Serial.begin(9600);
  PORTD = 0b00000100; //PA low, MOT high
  float PA_pretime_calc = 0;
  float PA_posttime_calc = 0;
  if (OperationMode == OffOff){
    //Serial.print("\n Wrong loop");
    PA_pretime_calc = PA_prefrac;
    PA_posttime_calc = PA_postfrac;
  }
  else if (OperationMode == OnOff){
    //Serial.print("\n Right loop");
    PA_pretime_calc = -1*PA_prefrac;
    PA_posttime_calc = PA_postfrac;
  }
  else if (OperationMode == OffOn){
    //Serial.print("\n Wrong loop");
    PA_pretime_calc = PA_prefrac;
    PA_posttime_calc = -1*PA_postfrac;
  }
  else if (OperationMode == OnOn){
    //Serial.print("\n Wrong loop");
    PA_pretime_calc = -1*PA_prefrac;
    PA_posttime_calc = -1*PA_postfrac;
  }
  float total_time_perc = MOT_uptime+PA_uptime+PA_pretime_calc+PA_posttime_calc;
  Serial.print("\n MOT_uptime: ");
  Serial.print(MOTtime);
  Serial.print("\n PA_uptime: ");
  Serial.print(PAtime);
  Serial.print("\n PA pretime: ");
  Serial.print(PA_pretime);
  Serial.print("\n PA posttime: ");
  Serial.print(PA_posttime);
  if (total_time_perc != 100){
    Serial.print("\n TOTAL TIME EXCEEDS 100%!!! CHECK TIMINGS \n");
  }
  else if (total_time_perc == 100){
    Serial.print("\n TOTAL TIME EQUALS 100%!!! LETS ROLL \n");
  }
  //Serial.print("\n Setup complete");

}

void loop(){
  
  float analog_signal = 0;
  
  digitalTrigger = digitalRead(triggerPin2);
  //Serial.print(digitalTrigger);
  //Serial.print("\n");
  //Serial.print("\n");
  //analog_signal = analogRead(triggerPin);
  PORTD = 0b00000100; //PA low, MOT high
  //Serial.print("\n First analog reading:");
  //Serial.print(analog_signal);
  while(digitalTrigger){
    //Serial.print("\n In Loop Analog signal: ");
    //Serial.print(micros());
    //Serial.print("\n");
    
      #if OperationMode == OffOff
        //Serial.print("\n Right loop");
        dutycycleoffoff();
      #elif OperationMode == OnOff
        //Serial.print("\n Wrong loop");
        dutycycleonoff();
      #elif OperationMode == OffOn
        //Serial.print("\n Wrong loop");
        dutycycleoffon();
      #elif OperationMode == OnOn
        //Serial.print("\n Wrong loop");
        dutycycleonon();
      #endif
    digitalTrigger = digitalRead(triggerPin2);
  }
}



/*
void loop() {  
  digitalWrite(MOTPin, HIGH);
  float analog_signal = 0;
  analog_signal = analogRead(triggerPin);
  //Serial.print("\n Analog signal: ");
  //Serial.print(analog_signal);
  if(analog_signal>50.0) {
    //Serial.print("\n In first duty cycle");
    digitalWrite(MOTPin, HIGH);
    //Serial.print("\n MOT On, PA Off");
    delay(MOTtime - PA_predelaytime);
    digitalWrite(PAPin, HIGH);
    //Serial.print("\n MOT On, PA On");
    delay(PA_predelaytime);
    digitalWrite(MOTPin, LOW);
    //Serial.print("\n MOT Off, PA On");
    delay(PAtime - PA_predelaytime - PA_postdelaytime);
    digitalWrite(MOTPin, HIGH);
    //Serial.print("\n MOT On, PA On");
    delay(PA_postdelaytime);
    digitalWrite(PAPin, LOW);
    //Serial.print("\n End of first duty cycle, MOT On PA Off");
  }
  while(analog_signal>50){
    //Serial.print("\n In Loop Analog signal: ");
    //Serial.print(analog_signal);
    rundutycycle(MOTtime, PAtime, PA_predelaytime, PA_postdelaytime);
    analog_signal = analogRead(triggerPin);
  }
  digitalWrite(MOTPin, HIGH);
  digitalWrite(PAPin, LOW);
}


void rundutycycle(float MT, float PT, float PTpre, float PTpost){
  digitalWrite(MOTPin, HIGH);  
  //PORTD = 0b00100100; //PORTD = PORTD | 0b00000100;
  delay(MT-PTpre-PTpost);
  digitalWrite(PAPin, HIGH);
  if(PTpre)
    delay(PTpre);
  digitalWrite(MOTPin, LOW);
  delay(PT-PTpre-PTpost);
  digitalWrite(MOTPin, HIGH);
  if(PTpost)
    delay(PTpost);
  digitalWrite(PAPin, LOW);
}*/



void dutycycleoffoff(){
  PORTD = 0b00000100; //PA low, MOT high
  delayMicroseconds(MOTtime);
  #if PA_prefrac != 0 
    PORTD = 0b00000000; //PA low, MOT low
    delayMicroseconds(PA_pretime);
  #endif
  PORTD = 0b00100000; //PA high, MOT low
  delayMicroseconds(PAtime);
  
  #if PA_postfrac != 0
    PORTD = 0b00000000; //PA low, MOT low
    delayMicroseconds(PA_posttime);
  #endif
}

void dutycycleonoff(){
  PORTD = 0b00000100; //PA low, MOT high
  delayMicroseconds(MOTtime);
  PORTD = 0b00100100; //PA high, MOT high
  #if PA_prefrac != 0
    delayMicroseconds(PA_pretime);
  #endif
  PORTD = 0b00100000; //PA high, MOT low
  delayMicroseconds(PAtime);
  PORTD = 0b00000000; //PA low, MOT low
  #if PA_postfrac != 0
    delayMicroseconds(PA_posttime);
  #endif
}

void dutycycleoffon(){
  PORTD = 0b00000100; //PA low, MOT high
  delayMicroseconds(MOTtime);
  PORTD = 0b00000000; //PA low, MOT low
  #if PA_prefrac != 0
    delayMicroseconds(PA_pretime);
  #endif
  PORTD = 0b00100000; //PA high, MOT low
  delayMicroseconds(PAtime);
  PORTD = 0b00100100; //PA high, MOT high
  #if PA_postfrac != 0
    delayMicroseconds(PA_posttime);
  #endif
}  

void dutycycleonon(){
  PORTD = 0b00000100; //PA low, MOT high
  delayMicroseconds(MOTtime);
  PORTD = 0b00100100; //PA high, MOT high
  #if PA_prefrac != 0
    delayMicroseconds(PA_pretime);
  #endif
  PORTD = 0b00100000; //PA high, MOT low
  delayMicroseconds(PAtime);
  PORTD = 0b00100100; //PA high, MOT high
  #if PA_postfrac != 0
    delayMicroseconds(PA_posttime);
  #endif
}

//PORTD = 0b00100100; //PA high, MOT high
//PORTD = 0b00000100; //PA low, MOT high
//PORTD = 0b00100000; //PA high, MOT low
//PORTD = 0b00000000; //PA low, MOT low


float simanalog(){
  // waits 5 seconds before returning a voltage of 1.5V
  delay(5000);
  return (1.5);
}

