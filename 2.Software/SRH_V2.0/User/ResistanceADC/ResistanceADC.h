#ifndef __RESISTANCEADC_H
#define __RESISTANCEADC_H

#include "main.h"

#define MaxAD 4095              //12位ADC
#define Res_resistance  8870     //定值电阻

void ADCStartOnce(void);     //开启一次ADC采集
void ResValueTran(void);     //转换成电阻值
void TemValueTran(void);     //转换成温度值

uint32_t processAndmeanFilterChannel(uint32_t adcValue, int channel);  // 添加新的采样值到数组，并进行均值滤波
uint32_t processAndmedianFilterChannel(uint32_t adcValue, int channel);   // 添加新的采样值到数组，并进行中值滤波


#endif


