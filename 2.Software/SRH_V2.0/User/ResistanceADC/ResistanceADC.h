#ifndef __RESISTANCEADC_H
#define __RESISTANCEADC_H

#include "main.h"

#define MaxAD 4095              //12λADC
#define Res_resistance  8870     //��ֵ����

void ADCStartOnce(void);     //����һ��ADC�ɼ�
void ResValueTran(void);     //ת���ɵ���ֵ
void TemValueTran(void);     //ת�����¶�ֵ

uint32_t processAndmeanFilterChannel(uint32_t adcValue, int channel);  // ����µĲ���ֵ�����飬�����о�ֵ�˲�
uint32_t processAndmedianFilterChannel(uint32_t adcValue, int channel);   // ����µĲ���ֵ�����飬��������ֵ�˲�


#endif


