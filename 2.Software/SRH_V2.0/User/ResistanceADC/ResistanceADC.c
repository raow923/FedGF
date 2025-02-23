#include "ResistanceADC.h"
#include "adc.h"
#include <math.h>
#include <stdbool.h>

uint32_t adDmaValue[4];
uint32_t adDmaRaw[4];
float Res[4];
float Tem[4];

//float F_max=20;         //ѹ�����ֵ
//float T_max=50;          //�¶����ֵ


#define SAMPLE_SIZE1 10   // �������ڴ�С
#define NUM_CHANNELS1 4   // ��·����
int samples1[NUM_CHANNELS1][SAMPLE_SIZE1] = {0}; // ��·����ֵ�洢����
int currentIndex1[NUM_CHANNELS1] = {0};   // ÿ·�ĵ�ǰ����ֵ����
// ����µĲ���ֵ�����飬�����о�ֵ�˲�
uint32_t processAndmeanFilterChannel(uint32_t adcValue, int channel)
{
	currentIndex1[channel] = (currentIndex1[channel] + 1) % SAMPLE_SIZE1; // �������������
	samples1[channel][currentIndex1[channel]] = adcValue; // �����µ�����

	// ��ֵ�˲�
	uint32_t sum = 0;
	for(int i = 0; i < SAMPLE_SIZE1; i++)
	{
		sum += samples1[channel][i];
	}
	return sum / SAMPLE_SIZE1;
}

#define SAMPLE_SIZE2 10   // �������ڴ�С
#define NUM_CHANNELS2 4   // ��·����
int samples2[NUM_CHANNELS2][SAMPLE_SIZE2]; // ��·����ֵ�洢����
int currentIndex2[NUM_CHANNELS2] = {0};   // ÿ·�ĵ�ǰ����ֵ����
// ����µĲ���ֵ�����飬��������ֵ�˲�
uint32_t processAndmedianFilterChannel(uint32_t adcValue, int channel)
{
	currentIndex2[channel] = (currentIndex2[channel] + 1) % SAMPLE_SIZE2; // �������������
	samples2[channel][currentIndex2[channel]] = adcValue; // �����µ�����

	// ��ֵ�˲�
	uint32_t window[SAMPLE_SIZE2];
	for(int i = 0; i < SAMPLE_SIZE2; i++)
	{
		window[i] = samples2[channel][i];
	}

	// �Դ����ڵ����ݽ�������
	for(int i = 0; i < SAMPLE_SIZE2 - 1; i++)
	{
		for (int j = i + 1; j < SAMPLE_SIZE2; j++)
		{
			if (window[i] > window[j])
				{
					int temp = window[i];
					window[i] = window[j];
					window[j] = temp;
				}
		}
	}

	// ��ֵ�Ǵ����м��ֵ
	return window[SAMPLE_SIZE2 / 2];
}

//ADCת��һ��
void ADCStartOnce()
{
	HAL_ADC_Start_DMA(&hadc1, (uint32_t*)&adDmaRaw, 4);
	//��ֵ�˲�
	for (int channel = 0; channel < NUM_CHANNELS1; channel++)
	{
		adDmaValue[channel] = processAndmeanFilterChannel(adDmaRaw[channel], channel);
	}
//	//��ֵ�˲�
//	 for (int channel = 0; channel < NUM_CHANNELS2; channel++)
//	{
//		adDmaValue[channel] = processAndmedianFilterChannel(adDmaRaw[channel], channel);
//	}
}
//��ֵ����
void ResValueTran()
{	
	Res[0] = (Res_resistance*adDmaValue[0])/(MaxAD-adDmaValue[0]);
	Res[1] = (Res_resistance*adDmaValue[1])/(MaxAD-adDmaValue[1]);
	Res[2] = (Res_resistance*adDmaValue[2])/(MaxAD-adDmaValue[2]);
	Res[3] = (Res_resistance*adDmaValue[3])/(MaxAD-adDmaValue[3]);
}
//ѹ�����¶ȼ���
void TemValueTran()
{
	ResValueTran();
//	Tem[0] = 126491106/pow(Res[0],1.5);  //��λ ��
	Tem[0] = -25.72*log(Res[0])+264.82;
	Tem[1] = -25.72*log(Res[1])+264.82;  //y = -25.72ln(x) + 264.82
	Tem[2] = pow(145050 / Res[2], 2.066);
	Tem[3] = pow(145050 / Res[3], 2.066);
	
}

