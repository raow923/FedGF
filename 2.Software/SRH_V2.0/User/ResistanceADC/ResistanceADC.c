#include "ResistanceADC.h"
#include "adc.h"
#include <math.h>
#include <stdbool.h>

uint32_t adDmaValue[4];
uint32_t adDmaRaw[4];
float Res[4];
float Tem[4];

//float F_max=20;         //压力最大值
//float T_max=50;          //温度最大值


#define SAMPLE_SIZE1 10   // 采样窗口大小
#define NUM_CHANNELS1 4   // 四路数据
int samples1[NUM_CHANNELS1][SAMPLE_SIZE1] = {0}; // 四路采样值存储数组
int currentIndex1[NUM_CHANNELS1] = {0};   // 每路的当前采样值索引
// 添加新的采样值到数组，并进行均值滤波
uint32_t processAndmeanFilterChannel(uint32_t adcValue, int channel)
{
	currentIndex1[channel] = (currentIndex1[channel] + 1) % SAMPLE_SIZE1; // 丢弃最早的数据
	samples1[channel][currentIndex1[channel]] = adcValue; // 引入新的数据

	// 均值滤波
	uint32_t sum = 0;
	for(int i = 0; i < SAMPLE_SIZE1; i++)
	{
		sum += samples1[channel][i];
	}
	return sum / SAMPLE_SIZE1;
}

#define SAMPLE_SIZE2 10   // 采样窗口大小
#define NUM_CHANNELS2 4   // 四路数据
int samples2[NUM_CHANNELS2][SAMPLE_SIZE2]; // 四路采样值存储数组
int currentIndex2[NUM_CHANNELS2] = {0};   // 每路的当前采样值索引
// 添加新的采样值到数组，并进行中值滤波
uint32_t processAndmedianFilterChannel(uint32_t adcValue, int channel)
{
	currentIndex2[channel] = (currentIndex2[channel] + 1) % SAMPLE_SIZE2; // 丢弃最早的数据
	samples2[channel][currentIndex2[channel]] = adcValue; // 引入新的数据

	// 中值滤波
	uint32_t window[SAMPLE_SIZE2];
	for(int i = 0; i < SAMPLE_SIZE2; i++)
	{
		window[i] = samples2[channel][i];
	}

	// 对窗口内的数据进行排序
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

	// 中值是窗口中间的值
	return window[SAMPLE_SIZE2 / 2];
}

//ADC转换一次
void ADCStartOnce()
{
	HAL_ADC_Start_DMA(&hadc1, (uint32_t*)&adDmaRaw, 4);
	//均值滤波
	for (int channel = 0; channel < NUM_CHANNELS1; channel++)
	{
		adDmaValue[channel] = processAndmeanFilterChannel(adDmaRaw[channel], channel);
	}
//	//中值滤波
//	 for (int channel = 0; channel < NUM_CHANNELS2; channel++)
//	{
//		adDmaValue[channel] = processAndmedianFilterChannel(adDmaRaw[channel], channel);
//	}
}
//阻值计算
void ResValueTran()
{	
	Res[0] = (Res_resistance*adDmaValue[0])/(MaxAD-adDmaValue[0]);
	Res[1] = (Res_resistance*adDmaValue[1])/(MaxAD-adDmaValue[1]);
	Res[2] = (Res_resistance*adDmaValue[2])/(MaxAD-adDmaValue[2]);
	Res[3] = (Res_resistance*adDmaValue[3])/(MaxAD-adDmaValue[3]);
}
//压力、温度计算
void TemValueTran()
{
	ResValueTran();
//	Tem[0] = 126491106/pow(Res[0],1.5);  //单位 克
	Tem[0] = -25.72*log(Res[0])+264.82;
	Tem[1] = -25.72*log(Res[1])+264.82;  //y = -25.72ln(x) + 264.82
	Tem[2] = pow(145050 / Res[2], 2.066);
	Tem[3] = pow(145050 / Res[3], 2.066);
	
}

